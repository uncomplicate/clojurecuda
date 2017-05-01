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
  be provided as strings (that may be stored in files) or binaries, written in CUDA C/C++.

  Where applicable, methods throw ExceptionInfo in case of errors thrown by the CUDA driver.
  "
  (:require [uncomplicate.commons
             [core :refer [Releaseable release wrap-float wrap-double wrap-long wrap-int]]
             [utils :refer [mask]]]
            [uncomplicate.clojurecuda
             [protocols :refer :all]
             [constants :refer :all]
             [info :refer [api-version cache-config limit* ctx-device]]
             [utils :refer [with-check]]]
            [clojure.string :as str]
            [clojure.core.async :refer [go >!]])
  (:import [jcuda Pointer NativePointerObject]
           [jcuda.driver JCudaDriver CUdevice CUcontext CUdeviceptr CUmemAttach_flags CUmodule
            CUfunction CUstream CUstream_flags CUresult CUstreamCallback CUevent CUevent_flags]
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

(extend-type Pointer
  WithOffset
  (with-offset [cu byte-offset]
   (.withByteOffset cu byte-offset)))

(extend-type CUdeviceptr
  Releaseable
  (release [dp]
    (with-check (JCudaDriver/cuMemFree dp) true))
  WithOffset
  (with-offset [cu byte-offset]
   (.withByteOffset cu byte-offset)))

(extend-type CUmodule
  Releaseable
  (release [m]
    (with-check (JCudaDriver/cuModuleUnload m) true)))

(extend-type CUstream
  Releaseable
  (release [s]
    (with-check (JCudaDriver/cuStreamDestroy s) true)))

(extend-type CUevent
  Releaseable
  (release [e]
    (with-check (JCudaDriver/cuEventDestroy e) true)))

(defn init
  "Initializes the CUDA driver."
  []
  (with-check (JCudaDriver/cuInit 0) true))

;; ================== Device Management ====================================

(defn device-count
  "Returns the number of CUDA devices on the system."
  ^long []
  (let [res (int-array 1)]
    (with-check (JCudaDriver/cuDeviceGetCount res) (aget res 0))))

(defn device
  "Returns a device specified with its ordinal number or string `id`"
  ([id]
   (let [res (CUdevice.)]
     (with-check
       (if (number? id)
         (JCudaDriver/cuDeviceGet res id)
         (JCudaDriver/cuDeviceGetByPCIBusId res id))
       res)))
  ([]
   (device 0)))

;; =================== Context Management ==================================

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

  Also see [cuCtxCreate](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html).
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

(defn current-context
  "Returns the CUDA context bound to the calling CPU thread.

  See [cuCtxGetCurrent](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html)
  "
  []
  (let [ctx (CUcontext.)]
    (with-check (JCudaDriver/cuCtxGetCurrent ctx) ctx)))

(defn current-context!
  "Binds the specified CUDA context to the calling CPU thread.

  See [cuCtxSetCurrent](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html)
  "
  [ctx]
  (with-check (JCudaDriver/cuCtxSetCurrent ctx) ctx))

(defn pop-context!
  "Pops the current CUDA context from the current CPU thread.

  See [cuCtxPopCurrent](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html)
  "
  []
  (let [ctx (CUcontext.)]
    (with-check (JCudaDriver/cuCtxPopCurrent ctx) ctx)))

(defn push-context!
  "Pushes a context on the current CPU thread.

  See [cuCtxPushCurrent](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html)
  "
  [ctx]
  (with-check (JCudaDriver/cuCtxPushCurrent ctx) ctx))

;; ================== Memory Management  ==============================================

(defn memcpy!
  "Copies `byte-count` or all possible device memory from `src` to `dst`.

  See [cuMemcpy](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html)"
  ([src dst ^long byte-count]
   (with-check (JCudaDriver/cuMemcpy (cu-ptr dst) (cu-ptr src) byte-count) dst))
  ([src dst]
   (memcpy! src dst (min (long (size src)) (long (size dst))))))

(defn memcpy-host!
  "Copies `byte-count` or all possible memory from `src` to `dst`, one of which
  has to be accessible from the host. If `hstream` is provided, the copy is asynchronous.
  Polymorphic function that figures out what needs to be done.

  See [cuMemcpyXtoY](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html)
  "
  ([src dst ^long byte-count hstream]
   (if hstream
     (memcpy-host* src dst byte-count hstream)
     (memcpy-host* src dst byte-count)))
  ([src dst arg]
   (if (integer? arg)
     (memcpy-host* src dst arg)
     (memcpy-host* src dst (min (long (size src)) (long (size dst))) arg)))
  ([src dst]
   (memcpy-host* src dst (min (long (size src)) (long (size dst))))))

(defn memset!
  "Sets `len` or all 32-bit segments of `cu-mem` to 32-bit integer `value`. If `hstream` is
  provided, does this asynchronously.

  See [cuMemset32D](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html)
  "
  ([cu-mem ^long value]
   (memset! cu-mem value (long (/ (long (size cu-mem)) Integer/BYTES))))
  ([cu-mem ^long value arg]
   (if (integer? arg)
     (with-check (JCudaDriver/cuMemsetD32 (cu-ptr cu-mem) value arg) cu-mem)
     (memset! cu-mem value (/ (long (size cu-mem)) Integer/BYTES) arg)))
  ([cu-mem ^long value ^long len hstream]
   (if hstream
     (with-check (JCudaDriver/cuMemsetD32Async (cu-ptr cu-mem) value len hstream) cu-mem)
     (memset! cu-mem value len))))

;; ==================== Linear memory ================================================

(deftype CULinearMemory [^CUdeviceptr cu ^Pointer p ^long s in-module]
  Releaseable
  (release [_]
    (if-not in-module (release cu) true))
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

(defn ^:private cu-linear-memory
  ([^CUdeviceptr cu ^long size in-module]
   (let [cu-arr (make-array CUdeviceptr 1)]
     (aset ^"[Ljcuda.driver.CUdeviceptr;" cu-arr 0 cu)
     (CULinearMemory. cu (Pointer/to ^"[Ljcuda.driver.CUdeviceptr;" cu-arr) size in-module)))
  ([^CUdeviceptr cu ^long size]
   (cu-linear-memory cu size false)))

(defn mem-alloc
  "Allocates the `size` bytes of memory on the device. Returns a [[CULinearMemory]] object.

  The memory is not cleared. `size` must be greater than `0`.

  See [cuMemAlloc](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html).
  "
  [^long size]
  (let [cu (CUdeviceptr.)]
    (with-check (JCudaDriver/cuMemAlloc cu size) (cu-linear-memory cu size))))

(defn mem-alloc-managed*
  "Allocates the `size` bytes of memory that will be automatically managed by the Unified Memory
  system, specified by an integer `flag`.

  Returns a [[CULinearmemory]] object.
  The memory is not cleared. `size` must be greater than `0`.

  See [cuMemAllocManaged](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html).
  "
  ([^long size ^long flag]
   (let [cu (CUdeviceptr.)]
     (with-check (JCudaDriver/cuMemAllocManaged cu size flag) (cu-linear-memory cu size)))))

(defn mem-alloc-managed
  "Allocates the `size` bytes of memory that will be automatically managed by the Unified Memory
  system, specified by a keyword `flag`.

  Returns a [[CULinearMemory]] object.
  Valid flags are: `:global`, `:host` and `:single` (the default).
  The memory is not cleared. `size` must be greater than `0`.

  See [cuMemAllocManaged](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html).
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

  See [cuMemHostAlloc](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html).
  "
  [^long size ^long flags]
  (let [p (Pointer.)]
    (with-check (JCudaDriver/cuMemHostAlloc p size flags) (cu-pinned-memory p size free-pinned))))

(defn mem-host-alloc
  "Allocates `size` bytes of page-locked, 'pinned' on the host, using keyword `flags`.
  For available flags, see [constants/mem-host-alloc-flags]

  Valid flags are: `:portable`, `:devicemap` and `:writecombined`. The default is none.
  The memory is not cleared. `size` must be greater than `0`.

  See [cuMemHostAlloc](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html).
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

  See [cuMemAllocHost](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html).
  "
  [^long size]
  (let [p (Pointer.)]
    (with-check (JCudaDriver/cuMemAllocHost p size) (cu-pinned-memory p size free-pinned))))

(defn mem-host-register*
  "Registers previously allocated Java `memory` structure and pins it, using raw integer `flags`.

   See [cuMemHostRegister](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html).
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

  See [cuMemHostRegister](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html).
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

;; ================== Module Management =====================================

(extend-type String
  ModuleLoad
  (module-load [data m]
    (with-check (JCudaDriver/cuModuleLoadData ^CUmodule m data) m)))

(extend-type (Class/forName "[B")
  ModuleLoad
  (module-load [binary m]
    (with-check (JCudaDriver/cuModuleLoadFatBinary ^CUmodule m ^bytes binary) m)))

(defn load!
  "Load a module's data from a [[ntrtc/ptx]] string, `nvrtcProgram`, or a binary `data`, for already
  existing module.

  See [cuModuleGetFunction](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html)
  "
  [^CUmodule m data]
  (module-load data m))

(defn module
  "Creates a new CUDA module and loads a string, `nvrtcProgram`, or a binary `data`."
  ([]
   (CUmodule.))
  ([data]
   (load! (CUmodule.) data)))

(defrecord GridDim [^long grid-x ^long grid-y ^long grid-z ^long block-x ^long block-y ^long block-z])

(defn grid-1d
  "Creates a 1-dimensional [[GridDim]] record with grid and block dimensions x."
  ([^long grid-x]
   (GridDim. grid-x 1 1 (min grid-x 256) 1 1))
  ([^long grid-x ^long block-x]
   (GridDim. grid-x 1 1 (min grid-x block-x) 1 1)))

(defn grid-2d
  "Creates a 2-dimensional [[GridDim]] record with grid and block dimensions x and y."
  ([^long grid-x ^long grid-y]
   (GridDim. grid-x grid-y 1 (min grid-x 256) 1 1))
  ([^long grid-x ^long grid-y ^long block-x]
   (GridDim. grid-x grid-y 1 (min grid-x block-x) 1 1))
  ([^long grid-x ^long grid-y ^long block-x ^long block-y]
   (GridDim. grid-x grid-y 1 (min grid-x block-x) (min grid-y block-y) 1)))

(defn grid-3d
  "Creates a 3-dimensional [[GridDim]] record with grid and block dimensions x, y, and z."
  ([^long grid-x ^long grid-y ^long grid-z]
   (GridDim. grid-x grid-y grid-z (min grid-x 256) 1 1))
  ([^long grid-x ^long grid-y ^long grid-z ^long block-x]
   (GridDim. grid-x grid-y grid-z (min grid-x block-x) 1 1))
  ([grid-x grid-y grid-z block-x block-y block-z]
   (GridDim. grid-x grid-y grid-z (min grid-x block-x) (min grid-y block-y) (min grid-z block-z))))

(defn global
  "Returns CUDA global `CULinearMemory` named `name` from module `m`, with optionally specified size..

  See [cuModuleGetFunction](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html)
  "
  [^CUmodule m name]
  (let [res (CUdeviceptr.)
        byte-size (long-array 1)]
    (with-check
      (JCudaDriver/cuModuleGetGlobal res byte-size m name)
      (cu-linear-memory res (aget byte-size 0) true))))

(defn parameters
  "Creates a `Pointer` to an array of `Pointer`s to CUDA `params`. `params` can be any object on
  device ([[CULinearMemory]] for example), or host (arrays, numbers) that makes sense as a kernel
  parameter per CUDA specification. Use the result as an argument in [[launch!]].
  "
  [& params]
  (Pointer/to ^"[Ljcuda.Pointer;" (into-array Pointer (map ptr params))))

;; ====================== Execution Control ==================================

(defn function
  "Returns CUDA kernel function named `name` from module `m`.

  See [cuModuleGetFunction](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html)
  "
  [m name]
  (let [res (CUfunction.)]
    (with-check (JCudaDriver/cuModuleGetFunction res m name) res)))

(defn launch!
  "Invokes the kernel `fun` on a grid-dim grid of blocks, using parameters `params`.

  Optionally, you can specify the amount of shared memory that will be available to each thread block,
  and `hstream` to use for execution.

  See [cuModuleGetFunction](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html)
  "
  ([^CUfunction fun ^GridDim grid-dim shared-mem-bytes hstream ^Pointer params]
   (with-check
     (JCudaDriver/cuLaunchKernel fun (.grid-x grid-dim) (.grid-y grid-dim) (.grid-z grid-dim)
                                 (.block-x grid-dim) (.block-y grid-dim) (.block-z grid-dim)
                                 shared-mem-bytes hstream params nil)
     fun))
  ([^CUfunction fun ^GridDim grid-dim hstream ^Pointer params]
   (launch! fun grid-dim 0 hstream params))
  ([^CUfunction fun ^GridDim grid-dim ^Pointer params]
   (launch! fun grid-dim 0 nil params)))

;; ================== Stream Management ======================================

(defn stream*
  "Create a stream using an optional `priority` and an integer `flag`.

  See [cuStreamCreate](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html)
  "
  ([^long flag]
   (let [res (CUstream.)]
     (with-check (JCudaDriver/cuStreamCreate res flag) res)))
  ([^long priority ^long flag]
   (let [res (CUstream.)]
     (with-check (JCudaDriver/cuStreamCreateWithPriority res flag priority) res))))

(defn stream
  "Create a stream using an optional integer `priority` and a keyword `flag`.

  Valid `flag`s are `:default` and `:non-blocking`.

  See [cuStreamCreate](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html)
  "
  ([]
   (stream* CUstream_flags/CU_STREAM_DEFAULT))
  ([flag]
   (stream* (or (stream-flags flag)
                (throw (ex-info "Invaling stream flag." {:flag flag :available stream-flags})))))
  ([^long priority flag]
   (stream* priority (or (stream-flags flag)
                         (throw (ex-info "Invaling stream flag." {:flag flag :available stream-flags}))))))

(defn ready?
  "Determine status (ready or not) of a compute stream or event.

  See [cuStreamQuery](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html),
  and [cuEventQuery](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html)
"
  [obj]
  (= CUresult/CUDA_SUCCESS (if (instance? CUstream obj)
                             (JCudaDriver/cuStreamQuery obj)
                             (JCudaDriver/cuEventQuery obj))))

(defn synchronize!
  "Block for the current context's or `stream`'s tasks to complete."
  ([]
   (with-check (JCudaDriver/cuCtxSynchronize) true))
  ([hstream]
   (with-check (JCudaDriver/cuStreamSynchronize hstream) hstream)))

(defrecord StreamCallbackInfo [^CUstream hstream status data])

(deftype StreamCallback [ch]
  CUstreamCallback
  (call [this hstream status data]
    (go (>! ch (->StreamCallbackInfo hstream (CUresult/stringFor status) data)))))

(defn callback
  "Creates a [[StreamCallback]] that writes [[StreamCallbackInfo]] into async channel `ch`."
  [ch]
  (StreamCallback. ch))

(defn add-callback!
  "Adds a [[StreamCallback]] to a compute stream, with optional `data` related to the call.

  See [cuStreamAddCallback](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html)"
  ([hstream ^StreamCallback callback data]
   (with-check (JCudaDriver/cuStreamAddCallback hstream callback data 0) hstream))
  ([hstream ^StreamCallback callback]
   (with-check (JCudaDriver/cuStreamAddCallback hstream callback nil 0) hstream)))

(defn wait-event!
  "Make a compute stream `hstream` wait on an event `ev

  See [cuStreamWaitEvent](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html)"
  [hstream ev]
  (with-check (JCudaDriver/cuStreamWaitEvent hstream ev 0) hstream))

;; ================== Event Management =======================================

(defn event*
  "Creates an event specified by integer `flags`.

  See [cuEventCreate](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html)
  "
  [^long flags]
  (let [res (CUevent.)]
    (with-check (JCudaDriver/cuEventCreate res flags) res)))

(defn event
  "Creates an event specified by keyword `flags`.

  Available flags are `:default`, `:blocking-sync`, `:disable-timing`, and `:interprocess`.

  See [cuEventCreate](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html)
  "
  ([]
   (event* CUevent_flags/CU_EVENT_DEFAULT))
  ([flag & flags]
   (event* (if flags
             (mask event-flags (cons flag flags))
             (or (event-flags flag)
                 (throw (ex-info "Unknown event flag." {:flag flag :available event-flags})))))))

(defn elapsed-time
  "Computes the elapsed time in milliseconds between `start-event` and `end-event`.

  See [cuEventElapsedTime](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html)
  "
  ^double [start-event end-event]
  (let [res (float-array 1)]
    (with-check (JCudaDriver/cuEventElapsedTime res start-event end-event) (aget res 0))))

(defn record!
  "Records an event `ev` on optional `stream`.

  See [cuEventRecord](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html)
  "
  ([stream event]
   (with-check (JCudaDriver/cuEventRecord event stream) stream))
  ([event]
   (with-check (JCudaDriver/cuEventRecord event nil) nil)))

;; ================== Peer Context Memory Access =============================

(defn can-access-peer
  "Queries if a device may directly access a peer device's memory.

  See [cuDeviceCanAccessPeer](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PEER__ACCESS.html)
  "
  [dev peer]
  (let [res (int-array 1)]
    (with-check (JCudaDriver/cuDeviceCanAccessPeer res dev peer) (pos? (aget res 0)))))

(defn p2p-attribute*
  "Queries attributes of the link between two devices.

  See [cuDeviceGetP2PAttribute](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PEER__ACCESS.html)
  "
  [dev peer ^long attribute]
  (let [res (int-array 1)]
    (with-check (JCudaDriver/cuDeviceGetP2PAttribute res attribute dev peer) (pos? (aget res 0)))))

(defn p2p-attribute
  "Queries attributes of the link between two devices.

  See [cuDeviceGetP2PAttribute](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PEER__ACCESS.html)
  "
  [dev peer attribute]
  (p2p-attribute* dev peer (or (p2p-attributes attribute)
                              (throw (ex-info "Unknown p2p attribute"
                                              {:attribute attribute :available p2p-attributes})))))

(defn disable-peer-access!
  "Disables direct access to memory allocations in a peer context and unregisters any registered allocations.

  See [cuCtxDisablePeerAccess](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PEER__ACCESS.html)
  "
  ([ctx]
   (let [res (int-array 1)]
     (with-check (JCudaDriver/cuCtxDisablePeerAccess ctx) ctx)))
  ([]
   (disable-peer-access! *context*)))

(defn enable-peer-access!
  "Enables direct access to memory allocations in a peer context and unregisters any registered allocations.

  See [cuCtxEnablePeerAccess](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PEER__ACCESS.html)
  "
  ([ctx]
   (let [res (int-array 1)]
     (with-check (JCudaDriver/cuCtxEnablePeerAccess ctx 0) ctx)))
  ([]
   (enable-peer-access! *context*)))
