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
             [core :refer [Releaseable release wrap-float wrap-double wrap-long wrap-int]]
             [utils :refer [mask]]]
            [uncomplicate.clojurecuda
             [constants :refer :all]
             [utils :refer [with-check with-check-nvrtc nvrtc-error]]]
            [clojure.string :as str]
            [clojure.core.async :refer [go >!]])
  (:import [jcuda Pointer NativePointerObject]
           [jcuda.driver JCudaDriver CUdevice CUcontext CUdeviceptr CUmemAttach_flags CUmodule
            CUfunction]
           [jcuda.nvrtc JNvrtc nvrtcProgram nvrtcResult]
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

(extend-type nvrtcProgram
  Releaseable
  (release [p]
    (with-check-nvrtc (JNvrtc/nvrtcDestroyProgram p) true)))

(extend-type CUmodule
  Releaseable
  (release [m]
    (with-check (JCudaDriver/cuModuleUnload m) true)))

(defn init
  "Initializes the CUDA driver."
  []
  (with-check (JCudaDriver/cuInit 0) true))

;; ================== Device ====================================

(defn device-count
  "Returns the number of CUDA devices on the system."
  ^long []
  (let [res (int-array 1)]
    (with-check (JCudaDriver/cuDeviceGetCount res) (aget res 0))))

(defn device
  "Returns a device specified with its ordinal number `ord`."
  ([^long ord]
   (let [res (CUdevice.)]
     (with-check (JCudaDriver/cuDeviceGet res ord) res)))
  ([]
   (device 0)))

;; =================== Context ==================================

(defn context*
  "Creates a CUDA context on the `device` using a raw integer `flag`.
  For available flags, see [[constants/ctx-flags]].
  "
  [dev ^long flags]
  (let [res (CUcontext.)]
    (with-check (JCudaDriver/cuCtxCreate res flags dev) res)))

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

(defn synchronize
  "TODO"
  []
  (with-check (JCudaDriver/cuCtxSynchronize) true))

;; ================== Memory ===================================

(defprotocol Mem
  "An object that represents memory that participates in CUDA operations.
  It can be on the device, or on the host.  Built-in implementations:
  cuda pointers, Java primitive arrays and ByteBuffers"
  (ptr [this]
    "`Pointer` to this object.")
  (size [this]
    "Memory size of this cuda or host object in bytes.")
  (memcpy-host* [this host size] [this host size hstream]))

(defprotocol DeviceMem
  (cu-ptr [this]
    "CUDA `CUdeviceptr` to this object."))

(defprotocol HostMem
  (host-ptr [this]
    "Host `Pointer` to this object.")
  (host-buffer [this]
    "The actual `ByteBuffer` on the host"))

;; ================== Polymorphic memcpy  ==============================================

(defn memcpy!
  ([src dst ^long byte-count]
   (with-check (JCudaDriver/cuMemcpy (cu-ptr dst) (cu-ptr src) byte-count) dst))
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

(deftype CULinearMemory [^CUdeviceptr cu ^Pointer p ^long s]
  Releaseable
  (release [_]
    (release cu))
  DeviceMem
  (cu-ptr [_]
    cu)
  Mem
  (ptr [_]
    p)
  (size [_]
    s)
  (memcpy-host* [this host byte-size]
    (with-check (JCudaDriver/cuMemcpyDtoH (host-ptr host) cu byte-size) host))
  (memcpy-host* [this host byte-size hstream]
    (with-check (JCudaDriver/cuMemcpyDtoHAsync (host-ptr host) cu byte-size hstream) host)))

(defn ^:private cu-linear-memory [^CUdeviceptr cu ^long size]
  (let [cu-arr (make-array CUdeviceptr 1)]
    (aset ^"[Ljcuda.driver.CUdeviceptr;" cu-arr 0 cu)
    (CULinearMemory. cu (Pointer/to ^"[Ljcuda.driver.CUdeviceptr;" cu-arr) size)))

(defn mem-alloc
  "Allocates the `size` bytes of memory on the device. Returns a [[CULinearmemory]] object.

  The memory is not cleared. `size` must be greater than `0`.

  See [cuMemAlloc](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1gb82d2a09844a58dd9e744dc31e8aa467).
  "
  [^long size]
  (let [cu (CUdeviceptr.)]
    (with-check (JCudaDriver/cuMemAlloc cu size) (cu-linear-memory cu size))))

(defn mem-alloc-managed*
  "Allocates the `size` bytes of memory that will be automatically managed by the Unified Memory
  system, specified by an integer `flag`.

  Returns a [[CULinearmemory]] object.
  The memory is not cleared. `size` must be greater than `0`.

  See [cuMemAllocManaged](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1gb347ded34dc326af404aa02af5388a32).
  "
  ([^long size ^long flag]
   (let [cu (CUdeviceptr.)]
     (with-check (JCudaDriver/cuMemAllocManaged cu size flag) (cu-linear-memory cu size)))))

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

(defn ^:private free-pinned [hp buf]
  (with-check (JCudaDriver/cuMemFreeHost hp) (release buf)))

(defn ^:private unregister-pinned [hp _]
  (with-check (JCudaDriver/cuMemHostUnregister hp) true))

(deftype CUPinnedMemory [^CUdeviceptr cu ^Pointer p ^Pointer hp ^ByteBuffer buf ^long s release-fn]
  Releaseable
  (release [_]
    (release-fn hp buf))
  DeviceMem
  (cu-ptr [_]
    cu)
  HostMem
  (host-ptr [_]
    hp)
  (host-buffer [_]
    buf)
  Mem
  (ptr [_]
    p)
  (size [_]
    s)
  (memcpy-host* [this host byte-size]
    (with-check (JCudaDriver/cuMemcpyDtoH (host-ptr host) cu byte-size) host))
  (memcpy-host* [this host byte-size hstream]
    (with-check (JCudaDriver/cuMemcpyDtoHAsync (host-ptr host) cu byte-size hstream) host)))

(defn ^:private cu-pinned-memory [^Pointer hp ^long size release-fn]
  (let [cu (CUdeviceptr.)]
    (with-check (JCudaDriver/cuMemHostGetDevicePointer cu hp 0)
      (let [cu-arr (make-array CUdeviceptr 1)
            buf (.order (.getByteBuffer hp 0 size) (ByteOrder/nativeOrder))]
        (aset ^"[Ljcuda.driver.CUdeviceptr;" cu-arr 0 cu)
        (CUPinnedMemory. cu (Pointer/to ^"[Ljcuda.driver.CUdeviceptr;" cu-arr) hp buf size release-fn)))))

(defn mem-host-alloc*
  "Allocates `size` bytes of page-locked, 'pinned' on the host, using raw integer `flags`.
  For available flags, see [constants/mem-host-alloc-flags]

  The memory is not cleared. `size` must be greater than `0`.

  See [cuMemHostAlloc](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g572ca4011bfcb25034888a14d4e035b9).
  "
  [^long size ^long flags]
  (let [p (Pointer.)]
    (with-check (JCudaDriver/cuMemHostAlloc p size flags) (cu-pinned-memory p size free-pinned))))

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
  (let [p (Pointer.)]
    (with-check (JCudaDriver/cuMemAllocHost p size) (cu-pinned-memory p size free-pinned))))

(defn mem-host-register*
  "Registers previously allocated Java `memory` structure and pins it, using raw integer `flags`.

   See [cuMemHostRegister](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1gf0a9fe11544326dabd743b7aa6b54223).
  "
  [memory ^long flags]
  (let [p ^Pointer (ptr memory)
        byte-size (size memory)]
    (with-check (JCudaDriver/cuMemHostRegister p byte-size flags)
      (cu-pinned-memory p byte-size unregister-pinned))))

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
(extend-type Float
  Mem
  (ptr [this]
    (ptr (wrap-float this))))

(extend-type Double
  Mem
  (ptr [this]
    (ptr (wrap-double this))))

(extend-type Integer
  Mem
  (ptr [this]
    (ptr (wrap-int this))))

(extend-type Long
  Mem
  (ptr [this]
    (ptr (wrap-long this))))

(extend-type (Class/forName "[F")
  HostMem
  (host-ptr [this]
    (ptr this))
  Mem
  (ptr [this]
    (Pointer/to ^floats this))
  (size [this]
    (* Float/BYTES (alength ^floats this)))
  (memcpy-host*
    ([this cu byte-size]
     (with-check (JCudaDriver/cuMemcpyHtoD (cu-ptr cu) (ptr this) byte-size) cu))
    ([this cu byte-size hstream]
     (with-check (JCudaDriver/cuMemcpyHtoDAsync (cu-ptr cu) (ptr this) byte-size hstream) cu))))

(extend-type (Class/forName "[D")
  HostMem
  (host-ptr [this]
    (ptr this))
  Mem
  (ptr [this]
    (Pointer/to ^doubles this))
  (size [this]
    (* Double/BYTES (alength ^doubles this)))
  (memcpy-host*
    ([this cu byte-size]
     (with-check (JCudaDriver/cuMemcpyHtoD (cu-ptr cu) (ptr this) byte-size) cu))
    ([this cu byte-size hstream]
     (with-check (JCudaDriver/cuMemcpyHtoDAsync (cu-ptr cu) (ptr this) byte-size hstream) cu))))

(extend-type (Class/forName "[I")
  HostMem
  (host-ptr [this]
    (ptr this))
  Mem
  (ptr [this]
    (Pointer/to ^ints this))
  (size [this]
    (* Integer/BYTES (alength ^ints this)))
  (memcpy-host*
    ([this cu byte-size]
     (with-check (JCudaDriver/cuMemcpyHtoD (cu-ptr cu) (ptr this) byte-size) cu))
    ([this cu byte-size hstream]
     (with-check (JCudaDriver/cuMemcpyHtoDAsync (cu-ptr cu) (ptr this) byte-size hstream) cu))))

(extend-type (Class/forName "[J")
  HostMem
  (host-ptr [this]
    (ptr this))
  Mem
  (ptr [this]
    (Pointer/to ^longs this))
  (size [this]
    (* Long/BYTES (alength ^longs this)))
  (memcpy-host*
    ([this cu byte-size]
     (with-check (JCudaDriver/cuMemcpyHtoD (cu-ptr cu) (ptr this) byte-size) cu))
    ([this cu byte-size hstream]
     (with-check (JCudaDriver/cuMemcpyHtoDAsync (cu-ptr cu) (ptr this) byte-size hstream) cu))))

(extend-type (Class/forName "[B")
  HostMem
  (host-ptr [this]
    (ptr this))
  Mem
  (ptr [this]
    (Pointer/to ^bytes this))
  (size [this]
    (alength ^bytes this))
  (memcpy-host*
    ([this cu byte-size]
     (with-check (JCudaDriver/cuMemcpyHtoD (cu-ptr cu) (ptr this) byte-size) cu))
    ([this cu byte-size hstream]
     (with-check (JCudaDriver/cuMemcpyHtoDAsync (cu-ptr cu) (ptr this) byte-size hstream) cu))))

(extend-type (Class/forName "[S")
  HostMem
  (hostptr [this]
    (ptr this))
  Mem
  (ptr [this]
    (Pointer/to ^shorts this))
  (size [this]
    (* Short/BYTES (alength ^shorts this)))
  (memcpy-host*
    ([this cu byte-size]
     (with-check (JCudaDriver/cuMemcpyHtoD (cu-ptr cu) (ptr this) byte-size) cu))
    ([this cu byte-size hstream]
     (with-check (JCudaDriver/cuMemcpyHtoDAsync (cu-ptr cu) (ptr this) byte-size hstream) cu))))

(extend-type (Class/forName "[C")
  HostMem
  (host-ptr [this]
    (ptr this))
  Mem
  (ptr [this]
    (Pointer/to ^chars this))
  (size [this]
    (* Character/BYTES (alength ^chars this)))
  (memcpy-host*
    ([this cu byte-size]
     (with-check (JCudaDriver/cuMemcpyHtoD (cu-ptr cu) (ptr this) byte-size) cu))
    ([this cu byte-size hstream]
     (with-check (JCudaDriver/cuMemcpyHtoDAsync (cu-ptr cu) (ptr this) byte-size hstream) cu))))

(extend-type ByteBuffer
  HostMem
  (host-ptr [this]
    (ptr this))
  Mem
  (ptr [this]
    (Pointer/toBuffer this))
  (size [this]
    (.capacity ^ByteBuffer this))
  (memcpy-host*
    ([this cu byte-size]
     (with-check (JCudaDriver/cuMemcpyHtoD (cu-ptr cu) (ptr this) byte-size) cu))
    ([this cu byte-size hstream]
     (with-check (JCudaDriver/cuMemcpyHtoDAsync (cu-ptr cu) (ptr this) byte-size hstream) cu))))

;; ====================== Nvrtc program JIT ========================================

(defn program*
  "TODO"
  [name source-code source-headers include-names]
  (let [res (nvrtcProgram.)]
    (with-check-nvrtc
      (JNvrtc/nvrtcCreateProgram res source-code name (count source-headers) source-headers include-names)
      res)))

(defn program
  "TODO"
  ([name source-code headers]
   (program* name source-code (into-array String (take-nth 2 (next headers)))
                    (into-array String (take-nth 2 headers))))
  ([name source-code]
   (program* name source-code nil nil))
  ([source-code]
   (program* nil source-code nil nil)))

(defn program-log
  "TODO"
  [^nvrtcProgram program]
  (let [res (make-array String 1)]
    (with-check-nvrtc (JNvrtc/nvrtcGetProgramLog program res) (aget ^objects res 0))))

(defn compile*
  "TODO"
  ([^nvrtcProgram program options]
   (let [err (JNvrtc/nvrtcCompileProgram program (count options) options)]
     (if (= nvrtcResult/NVRTC_SUCCESS err)
       program
       (throw (nvrtc-error err (program-log program)))))))

(defn compile!
  "TODO"
  ([^nvrtcProgram program options]
   (compile* program (into-array String options)))
  ([^nvrtcProgram program]
   (compile* program nil)))

(defn ptx
  "TODO"
  [^nvrtcProgram program]
  (let [res (make-array String 1)]
    (with-check-nvrtc (JNvrtc/nvrtcGetPTX program res) (aget ^objects res 0))))

;; ========================= Module ==============================================

(defn load-data!
  "TODO"
  [^CUmodule m data]
  (with-check (JCudaDriver/cuModuleLoadData m (str (if (instance? nvrtcProgram data) (ptx data) data))) m))

(defn module
  "TODO"
  ([]
   (CUmodule.))
  ([data]
   (load-data! (CUmodule.) data)))

;; ========================= Function ===========================================

(defrecord WorkSize [^long grid-x ^long grid-y ^long grid-z ^long block-x ^long block-y ^long block-z])

(defn work-size-1d
  "TODO"
  ([^long grid-x]
   (WorkSize. grid-x 1 1 256 1 1))
  ([^long grid-x ^long block-x]
   (WorkSize. grid-x 1 1 block-x 1 1)))

(defn work-size-2d
  "TODO"
  ([^long grid-x ^long grid-y]
   (WorkSize. grid-x grid-y 1 256 1 1))
  ([^long grid-x ^long grid-y ^long block-x]
   (WorkSize. grid-x grid-y 1 block-x 1 1))
  ([^long grid-x ^long grid-y ^long block-x ^long block-y]
   (WorkSize. grid-x grid-y 1 block-x block-y 1)))

(defn work-size-3d
  "TODO"
  ([^long grid-x ^long grid-y ^long grid-z]
   (WorkSize. grid-x grid-y grid-z 256 1 1))
  ([^long grid-x ^long grid-y ^long grid-z ^long block-x]
   (WorkSize. grid-x grid-y grid-z block-x 1 1))
  ([grid-x grid-y grid-z block-x block-y block-z]
   (WorkSize. grid-x grid-y grid-z block-x block-y block-z)))

(defn function
  "TODO"
  [^CUmodule m name]
  (let [res (CUfunction.)]
    (with-check (JCudaDriver/cuModuleGetFunction res m name) res)))

(defn parameters
  "TODO"
  [& params]
  (Pointer/to ^"[Ljcuda.Pointer;" (into-array Pointer (map ptr params))))

(defn launch!
  "TODO"
  ([^CUfunction fun ^long grid-size-x ^long block-size-x ^Pointer params]
   (with-check
     (JCudaDriver/cuLaunchKernel fun grid-size-x 1 1 block-size-x 1 1 0 nil params nil)
     fun))
  ([^CUfunction fun ^WorkSize work-size ^Pointer params]
   (with-check
     (JCudaDriver/cuLaunchKernel fun (.grid-x work-size) (.grid-y work-size) (.grid-z work-size)
                                 (.block-x work-size) (.block-y work-size) (.block-z work-size)
                                 0 nil params nil)
     fun)))
