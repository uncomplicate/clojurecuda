;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.clojurecuda.core
  "Core ClojureCUDA functions for CUDA **host** programming. The kernels should
  be provided as strings (that may be stored in files), written in CUDA C.

  Where applicable, methods throw ExceptionInfo in case of errors thrown by the CUDA driver.
  "
  (:require [uncomplicate.commons
             [core :refer [Releaseable release]]
             [utils :refer [mask]]]
            [uncomplicate.clojurecuda
             [constants :refer :all]
             [utils :refer [with-check with-check-arr]]]
            [clojure.string :as str]
            [clojure.core.async :refer [go >!]])
  (:import jcuda.Pointer
           [jcuda.driver JCudaDriver CUdevice CUcontext CUdeviceptr CUmemAttach_flags]
           [java.nio ByteBuffer ByteOrder]))

(def ^{:dynamic true
       :doc "Dynamic var for binding the default context."}
  *context*)

;; ==================== Release resources =======================

(extend-type CUcontext
  Releaseable
  (release [c]
    (with-check (JCudaDriver/cuCtxDestroy c) true)))

(extend-type CUdeviceptr
  Releaseable
  (release [dp]
    (with-check (JCudaDriver/cuMemFree dp) true)))

(defn init
  "Initializes the CUDA driver."
  []
  (with-check (JCudaDriver/cuInit 0) true))

;; ================== Device ====================================

(defn device-count
  "Returns the number of CUDA devices on the system."
  ^long []
  (let [res (int-array 1)
        err (JCudaDriver/cuDeviceGetCount res)]
    (with-check err (aget res 0))))

(defn device
  "Returns a device specified with its ordinal number `ord`."
  [^long ord]
  (let [res (CUdevice.)
        err (JCudaDriver/cuDeviceGet res ord)]
    (with-check err res)))

;; =================== Context ==================================

(defn context*
  "Creates a CUDA context on the `device` using a raw integer `flag`.
  For available flags, see [[constants/ctx-flags]].
  "
  [dev ^long flags]
  (let [res (CUcontext.)
        err (JCudaDriver/cuCtxCreate res flags dev)]
    (with-check err res)))

(defn context
  "Creates a CUDA context on the `device` using a keyword `flag`.

  Valid flags are: `:sched-aulto`, `:sched-spin`, `:sched-yield`, `:sched-blocking-sync`,
  `:map-host`, `:lmem-resize-to-max`. The default is none.
  Must be released after use.

  Also see [cuCtxCreate](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g65dc0012348bc84810e2103a40d8e2cf).
  "
  ([dev flag]
   (context* dev (or (ctx-flags flag)
                     (throw (ex-info "Unknown context flag." {:flag flag :available ctx-flags})))))
  ([dev]
   (context* dev 0)))

(defmacro with-context
  "Dynamically binds `context` to the default context [[*context*]], and evaluates the body with
  the binding. Releases the context in the `finally` block.

  Take care *not* to release that context again in some other place; JVM might crash.
  "
  [context & body]
  `(binding [*context* ~context]
     (try ~@body
          (finally (release *context*)))))

;; ================== Memory ===================================

(defprotocol Mem
  "An object that represents memory that participates in CUDA operations.
  It can be on the device, or on the host.  Built-in implementations:
  cuda pointers, Java primitive arrays and ByteBuffers"
  (size [this]
    "Memory size of this cuda or host object in bytes.")
  (memcpy-host* [this host size] [this host size hstream]))

(defprotocol CUMem
  "A wrapper for CUdeviceptr objects, that also holds a Pointer to the cuda memory
  object, context that created it, and size in bytes. It is useful in many
  functions that need that (redundant in Java) data because of the C background
  of CUDA functions."
  (cu-ptr [this]
    "The raw JCuda memory object."))

(defprotocol HostMem
  (ptr [this]
    "CUDA `Pointer` to this object.")
  (host-buffer [this]
    "The actual `ByteBuffer` on the host"))

;; ================== Polymorphic memcpy  ==============================================

(defn memcpy!
  ([src dst ^long byte-count]
   (with-check
     (JCudaDriver/cuMemcpy (cu-ptr dst) (cu-ptr src) byte-count)
     dst))
  ([src dst]
   (memcpy! src dst (min (long (size src)) (long (size dst))))))

(defn memcpy-host!
  ([src dst ^long byte-count hstream]
   (memcpy-host* src dst byte-count hstream))
  ([src dst arg]
   (if (integer? arg)
     (memcpy-host* src dst arg)
     (memcpy-host* src dst (min (long (size src)) (long (size dst))) arg)))
  ([src dst]
   (memcpy-host* src dst (min (long (size src)) (long (size dst))))))

;; ==================== Linear memory ================================================

(deftype CULinearMemory [^CUdeviceptr cu ^long s]
  Releaseable
  (release [_]
    (release cu))
  Mem
  (size [_]
    s)
  (memcpy-host* [this host byte-size]
    (with-check (JCudaDriver/cuMemcpyDtoH (ptr host) cu byte-size) host))
  (memcpy-host* [this host byte-size hstream]
    (with-check (JCudaDriver/cuMemcpyDtoHAsync (ptr host) cu byte-size hstream) host))
  CUMem
  (cu-ptr [_]
    cu))

(defn mem-alloc
  "Allocates the `size` bytes of memory on the device. Returns a [[CULinearmemory]] object.

  The memory is not cleared. `size` must be greater than `0`.

  See [cuMemAlloc](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1gb82d2a09844a58dd9e744dc31e8aa467).
  "
  [^long size]
  (let [res (CUdeviceptr.)
        err (JCudaDriver/cuMemAlloc res size)]
    (with-check err (->CULinearMemory res size))))

(defn mem-alloc-managed*
  "Allocates the `size` bytes of memory that will be automatically managed by the Unified Memory
  system, specified by an integer `flag`.

  Returns a [[CULinearmemory]] object.
  The memory is not cleared. `size` must be greater than `0`.

  See [cuMemAllocManaged](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1gb347ded34dc326af404aa02af5388a32).
  "
  ([^long size ^long flag]
   (let [res (CUdeviceptr.)
         err (JCudaDriver/cuMemAllocManaged res size flag)]
     (with-check err (->CULinearMemory res size)))))

(defn mem-alloc-managed
  "Allocates the `size` bytes of memory that will be automatically managed by the Unified Memory
  system, specified by a keyword `flag`.

  Returns a [[CULinearmemory]] object.
  Valid flags are: `:global`, `:host` and `:single` (the default).
  The memory is not cleared. `size` must be greater than `0`.

  See [cuMemAllocManaged](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1gb347ded34dc326af404aa02af5388a32).
  "
  ([^long size flag]
   (mem-alloc-managed* size (or (mem-attach-flags flag)
                                (throw (ex-info "Unknown mem-attach flag."
                                                {:flag flag :available mem-attach-flags})))))
  (^ByteBuffer [^long size]
   (mem-alloc-managed* size CUmemAttach_flags/CU_MEM_ATTACH_GLOBAL)))

;; =================== Pinned Memory ================================================

(defn ^:private free-pinned [p buf]
  (with-check (JCudaDriver/cuMemFreeHost p) (release buf)))

(defn ^:private unregister-pinned [p _]
  (with-check (JCudaDriver/cuMemHostUnregister p) true))

(deftype CUPinnedMemory [^CUdeviceptr cu ^Pointer p ^ByteBuffer buf ^long s release-fn]
  Releaseable
  (release [_]
    (release-fn p buf))
  HostMem
  (ptr [_]
    p)
  (host-buffer [_]
    buf)
  Mem
  (size [_]
    s)
  (memcpy-host* [this host byte-size]
    (with-check (JCudaDriver/cuMemcpyDtoH (ptr host) cu byte-size) host))
  (memcpy-host* [this host byte-size hstream]
    (with-check (JCudaDriver/cuMemcpyDtoHAsync (ptr host) cu byte-size hstream) host))
  CUMem
  (cu-ptr [_]
    cu))

(defn mem-host-alloc*
  "Allocates `size` bytes of page-locked, 'pinned' on the host, using raw integer `flags`.
  For available flags, see [constants/mem-host-alloc-flags]

  The memory is not cleared. `size` must be greater than `0`.

  See [cuMemHostAlloc](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g572ca4011bfcb25034888a14d4e035b9).
  "
  [^long size ^long flags]
  (let [p (Pointer.)
        err (JCudaDriver/cuMemHostAlloc p size flags)]
    (with-check err
      (let [cu (CUdeviceptr.)
            err (JCudaDriver/cuMemHostGetDevicePointer cu p 0)]
        (with-check err
          (->CUPinnedMemory cu p (.order (.getByteBuffer p 0 size) (ByteOrder/nativeOrder))
                            size free-pinned))))))

(defn mem-host-alloc
  "Allocates `size` bytes of page-locked, 'pinned' on the host, using keyword `flags`.
  For available flags, see [constants/mem-host-alloc-flags]

  Valid flags are: `:portable`, `:devicemap` and `:writecombined`. The default is none.
  The memory is not cleared. `size` must be greater than `0`.

  See [cuMemHostAlloc](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g572ca4011bfcb25034888a14d4e035b9).
  "
  ([^long size flags]
   (mem-host-alloc* size (if (keyword? flags)
                           (or (mem-host-alloc-flags flags)
                               (throw (ex-info "Unknown mem-host-alloc flag."
                                               {:flag flags :available mem-host-alloc-flags})))
                           (mask mem-host-alloc-flags flags))))
  ([^long size]
   (mem-host-alloc* size 0)))

(defn mem-alloc-host
  "Allocates `size` bytes of page-locked, 'pinned' on the host.

  The memory is not cleared. `size` must be greater than `0`.

  See [cuMemAllocHost](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1gdd8311286d2c2691605362c689bc64e0).
  "
  [^long size]
  (let [p (Pointer.)
        err (JCudaDriver/cuMemAllocHost p size)]
    (with-check err
      (let [cu (CUdeviceptr.)
            err (JCudaDriver/cuMemHostGetDevicePointer cu p 0)]
        (with-check err
          (->CUPinnedMemory cu p (.order (.getByteBuffer p 0 size) (ByteOrder/nativeOrder))
                            size free-pinned))))))

(defn mem-host-register*
  "Registers previously allocated Java `memory` structure and pins it, using raw integer `flags`.

   See [cuMemHostRegister](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1gf0a9fe11544326dabd743b7aa6b54223).
  "
  [memory ^long flags]
  (let [p ^Pointer (ptr memory)
        byte-size (size memory)
        err (JCudaDriver/cuMemHostRegister p byte-size flags)]
    (with-check err
      (let [cu (CUdeviceptr.)
            err (JCudaDriver/cuMemHostGetDevicePointer cu p 0)]
        (with-check err
          (->CUPinnedMemory cu p (.order (.getByteBuffer p 0 byte-size) (ByteOrder/nativeOrder))
                            byte-size unregister-pinned))))))

(defn mem-host-register
  "Registers previously allocated Java `memory` structure and pins it, using keyword `flags`.

  Valid flags are: `:portable`, and `:devicemap`. The default is none.
  The memory is not cleared.

  See [cuMemHostRegister](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1gf0a9fe11544326dabd743b7aa6b54223).
  "
  ([memory flags]
   (mem-host-register* memory (if (keyword? flags)
                                (or (mem-host-register-flags flags)
                                    (throw (ex-info "Unknown mem-host-register flag."
                                                    {:flag flags :available mem-host-register-flags})))
                                (mask mem-host-register-flags flags))))
  ([memory]
   (mem-host-register* memory 0)))

;; =============== Host memory  =================================

(extend-type (Class/forName "[F")
  HostMem
  (ptr [this]
    (Pointer/to ^floats this))
  Mem
  (size [this]
    (* Float/BYTES (alength ^floats this)))
  (memcpy-host*
    ([this cu byte-size]
     (with-check (JCudaDriver/cuMemcpyHtoD (cu-ptr cu) (ptr this) byte-size) cu))
    ([this cu byte-size hstream]
     (with-check (JCudaDriver/cuMemcpyHtoDAsync (cu-ptr cu) (ptr this) byte-size hstream) cu))))

(extend-type (Class/forName "[D")
  HostMem
  (ptr [this]
    (Pointer/to ^doubles this))
  Mem
  (size [this]
    (* Double/BYTES (alength ^doubles this)))
  (memcpy-host*
    ([this cu byte-size]
     (with-check (JCudaDriver/cuMemcpyHtoD (cu-ptr cu) (ptr this) byte-size) cu))
    ([this cu byte-size hstream]
     (with-check (JCudaDriver/cuMemcpyHtoDAsync (cu-ptr cu) (ptr this) byte-size hstream) cu))))

(extend-type (Class/forName "[I")
  HostMem
  (ptr [this]
    (Pointer/to ^ints this))
  Mem
  (size [this]
    (* Integer/BYTES (alength ^ints this)))
  (memcpy-host*
    ([this cu byte-size]
     (with-check (JCudaDriver/cuMemcpyHtoD (cu-ptr cu) (ptr this) byte-size) cu))
    ([this cu byte-size hstream]
     (with-check (JCudaDriver/cuMemcpyHtoDAsync (cu-ptr cu) (ptr this) byte-size hstream) cu))))

(extend-type (Class/forName "[J")
  HostMem
  (ptr [this]
    (Pointer/to ^longs this))
  Mem
  (size [this]
    (* Long/BYTES (alength ^longs this)))
  (memcpy-host*
    ([this cu byte-size]
     (with-check (JCudaDriver/cuMemcpyHtoD (cu-ptr cu) (ptr this) byte-size) cu))
    ([this cu byte-size hstream]
     (with-check (JCudaDriver/cuMemcpyHtoDAsync (cu-ptr cu) (ptr this) byte-size hstream) cu))))

(extend-type (Class/forName "[B")
  HostMem
  (ptr [this]
    (Pointer/to ^bytes this))
  Mem
  (size [this]
    (alength ^bytes this))
  (memcpy-host*
    ([this cu byte-size]
     (with-check (JCudaDriver/cuMemcpyHtoD (cu-ptr cu) (ptr this) byte-size) cu))
    ([this cu byte-size hstream]
     (with-check (JCudaDriver/cuMemcpyHtoDAsync (cu-ptr cu) (ptr this) byte-size hstream) cu))))

(extend-type (Class/forName "[S")
  HostMem
  (ptr [this]
    (Pointer/to ^shorts this))
  Mem
  (size [this]
    (* Short/BYTES (alength ^shorts this)))
  (memcpy-host*
    ([this cu byte-size]
     (with-check (JCudaDriver/cuMemcpyHtoD (cu-ptr cu) (ptr this) byte-size) cu))
    ([this cu byte-size hstream]
     (with-check (JCudaDriver/cuMemcpyHtoDAsync (cu-ptr cu) (ptr this) byte-size hstream) cu))))

(extend-type (Class/forName "[C")
  HostMem
  (ptr [this]
    (Pointer/to ^chars this))
  Mem
  (size [this]
    (* Character/BYTES (alength ^chars this)))
  (memcpy-host*
    ([this cu byte-size]
     (with-check (JCudaDriver/cuMemcpyHtoD (cu-ptr cu) (ptr this) byte-size) cu))
    ([this cu byte-size hstream]
     (with-check (JCudaDriver/cuMemcpyHtoDAsync (cu-ptr cu) (ptr this) byte-size hstream) cu))))

(extend-type ByteBuffer
  HostMem
  (ptr [this]
    (Pointer/toBuffer this))
  Mem
  (size [this]
    (.capacity ^ByteBuffer this))
  (memcpy-host*
    ([this cu byte-size]
     (with-check (JCudaDriver/cuMemcpyHtoD (cu-ptr cu) (ptr this) byte-size) cu))
    ([this cu byte-size hstream]
     (with-check (JCudaDriver/cuMemcpyHtoDAsync (cu-ptr cu) (ptr this) byte-size hstream) cu))))
