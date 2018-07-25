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
             [core :refer [with-release info]]
             [utils :refer [mask]]]
            [uncomplicate.clojurecuda.info :as cuda-info]
            [uncomplicate.clojurecuda.internal
             [protocols :refer :all]
             [constants :refer :all]
             [impl :refer :all]
             [utils :refer [with-check]]])
  (:import jcuda.Pointer
           [jcuda.driver JCudaDriver CUdevice CUdeviceptr CUmemAttach_flags CUmodule
            CUfunction CUstream CUstream_flags CUevent_flags CUlinkState]))

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
       {:device-id id}
       res)))
  ([]
   (device 0)))

;; =================== Context Management ==================================

(defn context
  "Creates a CUDA context on the `device` using a keyword `flag`.

  Valid flags are: `:sched-auto`, `:sched-spin`, `:sched-yield`, `:sched-blocking-sync`,
  `:map-host`, `:lmem-resize-to-max`. The default is none.
  Must be released after use.

  Also see [cuCtxCreate](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html).
  "
  ([dev flag]
   (wrap (context* dev (or (ctx-flags flag) (throw (ex-info "Unknown context flag."
                                                            {:flag flag :available ctx-flags}))))))
  ([dev]
   (wrap (context* dev 0))))

(defn current-context
  "Returns the CUDA context bound to the calling CPU thread.

  See [cuCtxGetCurrent](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html)
  "
  []
  (wrap (current-context*)))

(defn current-context!
  "Binds the specified CUDA context `ctx` to the calling CPU thread.

  See [cuCtxSetCurrent](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html)
  "
  [ctx]
  (current-context* (extract ctx))
  ctx)

(defn pop-context!
  "Pops the current CUDA context `ctx` from the current CPU thread.

  See [cuCtxPopCurrent](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html)
  "
  []
  (pop-context*))

(defn push-context!
  "Pushes a context `ctx` on the current CPU thread.

  See [cuCtxPushCurrent](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html)
  "
  [ctx]
  (push-context* (extract ctx))
  ctx)

(defmacro in-context
  "Pushes the context `ctx` to the top of the context stack, evaluates the body with `ctx`
  as the current context, and pops the context from the stack. Does NOT release the context.
  "
  [ctx & body]
  `(let [ctx# (extract ~ctx)]
     (push-context* ctx#)
     (try
       ~@body
       (finally (pop-context*)))))

(defmacro with-context
  "Pushes the context `ctx` to the top of the context stack, evaluates the body, and pops the context
  from the stack. Releases the context.
  "
  [ctx & body]
  `(with-release [ctx# ~ctx]
     (in-context ctx# ~@body)))

(defmacro with-default
  "Initializes CUDA, creates the default context and executes the body in it."
  [& body]
  `(do
     (init)
     (with-release [dev# (device)]
       (with-context (context dev#) ~@body))))

;; ================== Memory Management  ==============================================

(defn memcpy!
  "Copies `byte-count` or all possible device memory from `src` to `dst`. If `hstream` is supplied,
  executes asynchronously.

  See [cuMemcpy](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html)"
  ([src dst]
   (memcpy! src dst (min (long (size src)) (long (size dst)))))
  ([src dst count-or-stream]
   (if (number? count-or-stream)
     (with-check (JCudaDriver/cuMemcpy (extract dst) (extract src) count-or-stream) dst)
     (memcpy! src dst (min (long (size src)) (long (size dst))) count-or-stream)))
  ([src dst src-offset dst-offset count-or-stream]
   (if (number? count-or-stream)
     (with-check
       (JCudaDriver/cuMemcpy (with-offset dst dst-offset) (with-offset dst src-offset)
                             count-or-stream)
       dst)
     (memcpy! src dst src-offset dst-offset (min (long (size src)) (long (size dst)))
              count-or-stream)))
  ([src dst ^long byte-count hstream]
   (with-check
     (JCudaDriver/cuMemcpyAsync (extract dst) (extract src) byte-count (extract hstream))
     dst))
  ([src dst src-offset dst-offset byte-count hstream]
   (with-check
     (JCudaDriver/cuMemcpyAsync (with-offset dst dst-offset) (with-offset src src-offset)
                                byte-count (extract hstream))
     dst)))

(defn memcpy-host!
  "Copies `byte-count` or all possible memory from `src` to `dst`, one of which
  has to be accessible from the host. If `hstream` is provided, the copy is asynchronous.
  A polymorphic function that figures out what needs to be done.

  See [cuMemcpyXtoY](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html)
  "
  ([src dst ^long byte-count hstream]
   (if hstream
     (memcpy-host* src dst byte-count (extract hstream))
     (memcpy-host* src dst byte-count)))
  ([src dst arg]
   (if (integer? arg)
     (memcpy-host* src dst arg)
     (memcpy-host* src dst (min (long (size src)) (long (size dst))) (extract arg))))
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
     (with-check (JCudaDriver/cuMemsetD32 (extract cu-mem) value arg) cu-mem)
     (memset! cu-mem value (/ (long (size cu-mem)) Integer/BYTES) arg)))
  ([cu-mem ^long value ^long len hstream]
   (if hstream
     (with-check (JCudaDriver/cuMemsetD32Async (extract cu-mem) value len (extract hstream)) cu-mem)
     (memset! cu-mem value len))))

;; ==================== Linear memory ================================================

(defn mem-alloc
  "Allocates the `size` bytes of memory on the device.

  The old memory content is not cleared. `size` must be greater than `0`.

  See [cuMemAlloc](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html).
  "
  [^long size]
  (let [cu (CUdeviceptr.)]
    (with-check (JCudaDriver/cuMemAlloc cu size) (cu-linear-memory cu size))))

(defn mem-sub-region
  "Creates a [[CULinearMemory]] that references a sub-region of `mem` from origin to len."
  [mem ^long origin ^long byte-count]
  (let [origin (max 0 origin)
        byte-count (min byte-count (- (long (size mem)) origin))])
  (cu-linear-memory (with-offset mem origin) byte-count false))

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
  ([^long size]
   (mem-alloc-managed* size CUmemAttach_flags/CU_MEM_ATTACH_GLOBAL)))

;; =================== Pinned Memory ================================================

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
  (mem-alloc-host* size))

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

;; ================== Module Management =====================================

(defn link
  "Invokes CUDA linker on data provided as a vector `[[type source <options> <name>], ...]`.
  Produces a cubin compiled for particular architecture

  See [cuLinkCreate](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html) and
  related `likadd` functions.
  "
  ([data options]
   (wrap (link* (CUlinkState.) data options)))
  ([data]
   (link data nil))
  ([]
   (wrap (CUlinkState.))))

(defn link-complete
  [link-state]
  (link-complete* (extract link-state)))

(defn load!
  "Load a module's data from a [[ptx]] string, `nvrtcProgram`, java path, or a binary `data`,
  for already existing module.

  See [cuModuleGetFunction](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html)
  "
  ([m data]
   (module-load* data (extract m))
   m)
  ([m data options]
   (module-load-data-jit* (extract m) data (enc-jit-options options))
   m))

(defn module
  "Creates a new CUDA module and loads a string, nvrtc program, or binary `data`."
  ([]
   (wrap (CUmodule.)))
  ([data]
   (load! (module) data))
  ([data options]
   (load! (module) data options)))

(defrecord GridDim [^long grid-x ^long grid-y ^long grid-z ^long block-x ^long block-y ^long block-z])

(defn blocks-count
  "Computes the number of blocks that are needed for the global size kernel execution. "
  (^long [^long block-size ^long global-size]
   (if (< block-size global-size)
     (quot (+ global-size (dec block-size)) block-size)
     1))
  (^long [^long global-size]
   (blocks-count 1024 global-size)))

(defn grid-1d
  "Creates a 1-dimensional [[GridDim]] record with grid and block dimensions x.
  Note: dim-x is the total number of threads globally, not the number of blocks."
  ([^long dim-x]
   (let [block-x (min dim-x 1024)]
     (grid-1d dim-x block-x)))
  ([^long dim-x ^long block-x]
   (let [block-x (min dim-x block-x)]
     (GridDim. (blocks-count block-x dim-x) 1 1 block-x 1 1))))

(defn grid-2d
  "Creates a 2-dimensional [[GridDim]] record with grid and block dimensions x and y.
  Note: dim-x is the total number of threads globally, not the number of blocks."
  ([^long dim-x ^long dim-y]
   (let [block-x (min dim-x 32)
         block-y (min dim-y (long (/ 1024 block-x)))]
     (grid-2d dim-x dim-y block-x block-y)))
  ([^long dim-x ^long dim-y ^long block-x ^long block-y]
   (let [block-x (min dim-x block-x)
         block-y (min dim-y block-y)]
     (GridDim. (blocks-count block-x dim-x) (blocks-count block-y dim-y) 1 block-x block-y 1))))

(defn grid-3d
  "Creates a 3-dimensional [[GridDim]] record with grid and block dimensions x, y, and z.
  Note: dim-x is the total number of threads globally, not the number of blocks."
  ([^long dim-x ^long dim-y ^long dim-z]
   (let [block-x (min dim-x 32)
         block-y (min dim-y (long (/ 1024 block-x)))
         block-z (min dim-z (long (/ 1024 (* block-x block-y))))]
     (grid-3d dim-x dim-y dim-z block-x block-y block-z)))
  ([dim-x dim-y dim-z block-x block-y block-z]
   (let [block-x (min (long dim-x) (long block-x))
         block-y (min (long dim-y) (long block-y))
         block-z (min (long dim-z) (long block-z))]
     (GridDim. (blocks-count block-x dim-x) (blocks-count block-y dim-y)
               (blocks-count block-z dim-z) block-x block-y block-z))))

(defn global
  "Returns CUDA global linear memory named `name` from module `m`, with optionally specified size.

  See [cuModuleGetFunction](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html)
  "
  [m name]
  (global* (extract m) name))

(defn make-parameters
  "Creates an array of JCuda `Pointer`s."
  [^long len]
  (make-array Pointer len))

(defn set-parameter!
  "Sets the `i`th parameter in a parameter array `arr`"
  [^"[Ljcuda.Pointer;" arr ^long i parameter]
  (aset arr i (ptr parameter))
  arr)

(defn set-parameters!
  "Sets the `i`th parameter in a parameter array `arr` and the rest of `parameters` in places after `i`."
  [^"[Ljcuda.Pointer;" arr i parameter & parameters]
  (aset arr (long i) (ptr parameter))
  (loop [i (inc (long i)) parameters parameters]
    (if parameters
      (do
        (aset arr i (ptr (first parameters)))
        (recur (inc i) (next parameters)))
      arr)))

(defn parameters
  "Creates an array of `Pointer`s to CUDA `params`. `params` can be any object on
  device ([[CULinearMemory]] for example), or host (arrays, numbers) that makes sense as a kernel
  parameter per CUDA specification. Use the result as an parameterument in [[launch!]].
  "
  ([parameter & parameters]
   (let [len (if parameters (inc (count parameters)) 1)
         param-arr (make-parameters len)]
     (apply set-parameters! param-arr 0 parameter parameters))))

;; ====================== Execution Control ==================================

(defn function
  "Returns CUDA kernel function named `name` located in module `m`.

  See [cuModuleGetFunction](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html)
  "
  [m name]
  (let [res (CUfunction.)]
    (with-check (JCudaDriver/cuModuleGetFunction res (extract m) name) {:name name} res)))

(defn launch!
  "Invokes the kernel `fun` on a grid-dim grid of blocks, using parameters `params`.

  Optionally, you can specify the amount of shared memory that will be available to each thread block,
  and `hstream` to use for execution.

  See [cuModuleGetFunction](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html)
  "
  ([^CUfunction fun ^GridDim grid-dim shared-mem-bytes hstream ^"[Ljcuda.Pointer;" params]
   (with-check
     (JCudaDriver/cuLaunchKernel fun (.grid-x grid-dim) (.grid-y grid-dim) (.grid-z grid-dim)
                                 (.block-x grid-dim) (.block-y grid-dim) (.block-z grid-dim)
                                 shared-mem-bytes (extract hstream) (Pointer/to params) nil)
     {:kernel (info fun) :grid-dim grid-dim :hstream (info hstream)}
     hstream))
  ([^CUfunction fun ^GridDim grid-dim hstream params]
   (launch! fun grid-dim 0 hstream params))
  ([^CUfunction fun ^GridDim grid-dim params]
   (launch! fun grid-dim 0 nil params)))

;; ================== Stream Management ======================================

(defn stream
  "Create a stream using an optional integer `priority` and a keyword `flag`.

  Valid `flag`s are `:default` and `:non-blocking`.

  See [cuStreamCreate](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html)
  "
  ([]
   (wrap (stream* CUstream_flags/CU_STREAM_DEFAULT)))
  ([flag]
   (wrap (stream* (or (stream-flags flag)
                      (throw (ex-info "Invalid stream flag." {:flag flag :available stream-flags}))))))
  ([^long priority flag]
   (wrap (stream* priority (or (stream-flags flag)
                               (throw (ex-info  "Invaling stream flag."
                                                {:flag flag :available stream-flags})))))))

(def default-stream
  ^{:const true
    :doc "The default per-thread stream"}
  (wrap JCudaDriver/CU_STREAM_PER_THREAD))

(defn ready?
  "Determines status (ready or not) of a compute stream or event `obj`.

  See [cuStreamQuery](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html),
  and [cuEventQuery](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html)
  "
  [obj]
  (ready* (extract obj)))

(defn synchronize!
  "Block for the current context's or `stream`'s tasks to complete."
  ([]
   (synchronize*))
  ([hstream]
   (synchronize* (extract hstream))
   hstream))

(defn callback
  "Creates a stream callback that writes stream callback info into async channel `ch`.
  Available keys in callback info are `:status` and `:data`."
  [ch]
  (->StreamCallback ch))

(defn add-callback!
  "Adds a [[callback]] to a compute stream, with optional `data` related to the call.
  If `data` is not provided, places `hstream` under data.

  See [cuStreamAddCallback](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html)"
  ([hstream callback data]
   (add-callback* (extract hstream) callback data)
   hstream)
  ([hstream callback]
   (add-callback* (extract hstream) callback hstream)
   hstream))

(defn wait-event!
  "Makes a compute stream `hstream` wait on an event `ev`.

  See [cuStreamWaitEvent](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html)"
  [hstream ev]
  (wait-event* (extract hstream) (extract ev))
  hstream)

;; ================== Event Management =======================================

(defn event
  "Creates an event specified by keyword `flags`.

  Available flags are `:default`, `:blocking-sync`, `:disable-timing`, and `:interprocess`.

  See [cuEventCreate](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html)
  "
  ([]
   (wrap (event* CUevent_flags/CU_EVENT_DEFAULT)))
  ([flag & flags]
   (wrap (event* (if flags
                   (mask event-flags (cons flag flags))
                   (or (event-flags flag)
                       (throw (ex-info  "Unknown event flag." {:flag flag :available event-flags}))))))))

(defn elapsed-time
  "Computes the elapsed time in milliseconds between `start-event` and `end-event`.

  See [cuEventElapsedTime](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html)
  "
  ^double [start-event end-event]
  (elapsed-time* (extract start-event) (extract end-event)))

(defn record!
  "Records an event `ev` on optional `stream`.

  See [cuEventRecord](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html)
  "
  ([stream event]
   (record* (extract stream) (extract event))
   stream)
  ([event]
   (record* (extract event))))

;; ================== Peer Context Memory Access =============================

(defn can-access-peer
  "Queries if a device may directly access a peer device's memory.

  See [cuDeviceCanAccessPeer](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PEER__ACCESS.html)
  "
  [dev peer]
  (let [res (int-array 1)]
    (with-check (JCudaDriver/cuDeviceCanAccessPeer res dev peer) (pos? (aget res 0)))))

(defn p2p-attribute
  "Queries attributes of the link between two devices.

  See [cuDeviceGetP2PAttribute](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PEER__ACCESS.html)
  "
  [dev peer attribute]
  (p2p-attribute* dev peer (or (p2p-attributes attribute)
                               (throw (ex-info "Unknown p2p attribute"
                                               {:attribute attribute :available p2p-attributes})))))

(defn disable-peer-access!
  "Disables direct access to memory allocations in a peer context and unregisters
  any registered allocations.

  See [cuCtxDisablePeerAccess](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PEER__ACCESS.html)
  "
  ([ctx]
   (disable-peer-access* (extract ctx))
   ctx)
  ([]
   (disable-peer-access! (current-context))))

(defn enable-peer-access!
  "Enables direct access to memory allocations in a peer context and unregisters
  any registered allocations.

  See [cuCtxEnablePeerAccess](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PEER__ACCESS.html)
  "
  ([ctx]
   (enable-peer-access* (extract ctx))
   ctx)
  ([]
   (enable-peer-access! (current-context))))

;; ====================== Nvrtc program JIT ========================================

(defn program
  "Creates a CUDA program with an optional name from the `source-code`, and an optional
  hash map of headers (as strings) and their names."
  ([name source-code headers]
   (wrap (program* name source-code (into-array String (vals headers))
                   (into-array String (keys headers)))))
  ([source-code headers]
   (program nil source-code headers))
  ([source-code]
   (program nil source-code nil)))

(defn program-log
  "Returns the log string generated by the previous compilation of `prog`."
  [prog]
  (program-log* (extract prog)))

(defn compile!
  "Compiles the given `prog` using a list of string `options`."
  ([prog options]
   (compile* (extract prog) (into-array String options))
   prog)
  ([prog]
   (compile! prog nil)))

(defn ptx
  "Returns the PTX generated by the previous compilation of `prog`."
  ^String [prog]
  (ptx* (extract prog)))
