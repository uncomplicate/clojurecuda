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
             [core :refer [with-release let-release info wrap extract bytesize]]
             [utils :refer [mask count-groups dragan-says-ex]]]
            [uncomplicate.clojure-cpp
             :refer [null? pointer byte-pointer string-pointer int-pointer long-pointer size-t-pointer
                     pointer-pointer get-entry put-entry! element-count safe type-pointer
                     capacity!]]
            [uncomplicate.clojurecuda.info :as cuda-info]
            [uncomplicate.clojurecuda.internal
             [constants :refer :all]
             [impl :refer :all]
             [utils :refer [with-check]]])
  (:import [org.bytedeco.javacpp LongPointer SizeTPointer PointerPointer]
           org.bytedeco.cuda.global.cudart
           [org.bytedeco.cuda.cudart CUctx_st CUlinkState_st CUmod_st CUfunc_st CUstream_st CUevent_st]))

(defn init
  "Initializes the CUDA driver."
  []
  (with-check (cudart/cuInit 0) true))

;; ================== Device Management ====================================

(defn device-count
  "Returns the number of CUDA devices on the system."
  ^long []
  (let [res (int-pointer 1)]
    (with-check (cudart/cuDeviceGetCount res) (get-entry res 0))))

(defn device
  "Returns a device specified with its ordinal number or string `id`"
  ([id]
   (with-release [res (int-pointer 1)]
     (with-check
       (if (number? id)
         (cudart/cuDeviceGet res ^long id)
         (cudart/cuDeviceGetByPCIBusId res ^String id))
       {:device-id id}
       (->CUDevice (get-entry res 0)))))
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
   (context* (extract dev)
             (or (ctx-flags flag)
                 (throw (ex-info "Unknown context flag." {:flag flag :available ctx-flags})))))
  ([dev]
   (context* (extract dev) 0)))

(defn current-context
  "Returns the CUDA context bound to the calling CPU thread.

  See [cuCtxGetCurrent](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html)
  "
  []
  (current-context*))

(defn current-context!
  "Binds the specified CUDA context `ctx` to the calling CPU thread.

  See [cuCtxSetCurrent](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html)
  "
  [ctx]
  (current-context* ctx)
  ctx)

(defn pop-context!
  "Pops the current CUDA context `ctx` from the current CPU thread.

  See [cuCtxPopCurrent](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html)
  "
  []
  (let [ctx (CUctx_st.)]
    (with-check (cudart/cuCtxPopCurrent ctx) ctx)))

(defn push-context!
  "Pushes a context `ctx` on the current CPU thread.

  See [cuCtxPushCurrent](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html)
  "
  [^CUctx_st ctx]
  (with-check (cudart/cuCtxPushCurrent ctx) ctx))

(defmacro in-context
  "Pushes the context `ctx` to the top of the context stack, evaluates the body with `ctx`
  as the current context, and pops the context from the stack. Does NOT release the context.
  "
  [ctx & body]
  `(try
     (push-context! ~ctx)
     ~@body
     (finally (pop-context!))))

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
       (with-context (context dev#)
         ~@body))))

;; ================== Memory Management  ==============================================

(defn check-size [ptr ^long offset ^long byte-count]
  (when-not (and (<= 0 offset (+ offset byte-count) (bytesize ptr)))
    (dragan-says-ex "Requested bytes are out of the bounds of this device pointer."
                    {:offset offset :requested byte-count :available (bytesize ptr)})))

(defn memcpy!
  "Copies `byte-count` or all possible device memory from `src` to `dst`. If `hstream` is supplied,
  executes asynchronously.

  See [cuMemcpy](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html)"
  ([src dst]
   (memcpy! src dst 0 0 (min (bytesize src) (bytesize dst)) nil))
  ([src dst count-or-stream]
   (memcpy! src dst 0 0 count-or-stream))
  ([src dst src-offset dst-offset count-or-stream]
   (if (number? count-or-stream)
     (memcpy! src dst src-offset dst-offset count-or-stream nil)
     (memcpy! src dst src-offset dst-offset (min (bytesize src) (bytesize dst)) count-or-stream)))
  ([src dst ^long byte-count hstream]
   (memcpy! src dst 0 0 byte-count hstream))
  ([src dst src-offset dst-offset byte-count hstream]
   (check-size src src-offset byte-count)
   (check-size dst dst-offset byte-count)
   (with-check
     (if hstream
       (cudart/cuMemcpyAsync (offset dst dst-offset) (offset src src-offset) byte-count hstream)
       (cudart/cuMemcpy (offset dst dst-offset) (offset src src-offset) byte-count))
     dst)))

(defn memcpy-host!
  "Copies `byte-count` or all possible memory from `src` to `dst`, one of which
  has to be accessible from the host. If `hstream` is provided, the copy is asynchronous.
  A polymorphic function that figures out what needs to be done.

  See [cuMemcpyXtoY](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html)
  "
  ([src dst ^long byte-count hstream]
   (check-size src 0 byte-count)
   (check-size dst 0 byte-count)
   (if hstream
     (memcpy-host* dst src byte-count hstream)
     (memcpy-host* dst src byte-count))
   dst)
  ([src dst count-or-stream]
   (if (integer? count-or-stream)
     (memcpy-host! dst src count-or-stream nil)
     (memcpy-host* dst src (min (bytesize src) (bytesize dst)) count-or-stream))
   dst)
  ([src dst]
   (memcpy-host* dst src (min (bytesize src) (bytesize dst)))
   dst))

;; TODO implement a memset* protocol for all primitives.
(defn memset!
  "Sets `len` or all 32-bit segments of `dptr` to 32-bit integer `value`. If `hstream` is
  provided, does this asynchronously.

  See [cuMemset32D](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html)
  "
  ([dptr ^long value]
   (memset! dptr value (long (/ (long (bytesize dptr)) Integer/BYTES))))
  ([dptr ^long value arg]
   (if (integer? arg)
     (with-check (cudart/cuMemsetD32 (extract dptr) value arg) dptr)
     (memset! dptr value (/ (long (bytesize dptr)) Integer/BYTES) arg)))
  ([dptr ^long value ^long len hstream]
   (if hstream
     (with-check (cudart/cuMemsetD32Async (extract dptr) value len hstream) dptr)
     (memset! dptr value len))))

;; ==================== Linear memory ================================================

(defn mem-alloc
  "Allocates the `size` bytes of memory on the device.

  The old memory content is not cleared. `size` must be greater than `0`.

  See [cuMemAlloc](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html).
  "
  [^long size]
  (let-release [dptr (long-pointer 1)]
    (with-check (cudart/cuMemAlloc ^LongPointer dptr size)
      (->CUDevicePtr dptr size true))))

(defn mem-sub-region
  "Creates a [[CUDevicePtr]] that references a sub-region of `mem` from origin to `byte-count`."
  [mem ^long origin ^long byte-count]
  (check-size mem origin byte-count)
  (let-release [sub-dptr (long-pointer 1)]
    (->CUDevicePtr (put-entry! sub-dptr 0 (offset mem origin)) byte-count false)))

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
   (mem-alloc-managed* size cudart/CU_MEM_ATTACH_GLOBAL)))

;; =================== Pinned Memory ================================================

(defn mem-alloc-pinned
  "Allocates `size` bytes of page-locked, 'pinned' on the host, using keyword `flags`.
  For available flags, see [constants/mem-host-alloc-flags]

  Valid flags are: `:portable`, `:devicemap` and `:writecombined`. The default is none.
  The memory is not cleared. `size` must be greater than `0`.

  See [cuMemHostAlloc](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html).
  "
  ([^long size]
   (mem-host-alloc* (max 0 size) 0))
  ([^long size flags]
   (if-let [t (type-pointer flags)]
     (mem-host-alloc* (max 0 size) 0 t)
     (mem-host-alloc* (max 0 size)
                      (if (keyword? flags)
                        (or (mem-host-alloc-flags flags)
                            (throw (ex-info "Unknown mem-host-alloc flag."
                                            {:flag flags :available mem-host-alloc-flags})))
                        (mask mem-host-alloc-flags flags)))))
  ([^long size type flags]
   (if-let [t (type-pointer type)]
     (mem-host-alloc* (max 0 size)
                      (if (keyword? flags)
                        (or (mem-host-alloc-flags flags)
                            (throw (ex-info "Unknown mem-host-alloc flag."
                                            {:flag flags :available mem-host-alloc-flags})))
                        (mask mem-host-alloc-flags flags))
                      t)
     (throw (ex-info (format "Unknown data type: %s." (str type)))))))

(defn mem-register-pinned
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

(defn mem-alloc-mapped
  "TODO"
  ([^long size]
   (mem-alloc-host* (max 0 size)))
  ([^long size type]
   (mem-alloc-host* (max 0 size) (type-pointer type))))

;; ================== Module Management =====================================

(defn link
  "Invokes CUDA linker on data provided as a vector `[[type source <options> <name>], ...]`.
  Produces a cubin compiled for particular architecture

  See [cuLinkCreate](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html) and
  related `likadd` functions.
  "
  ([data options]
   (let-release [res (CUlinkState_st.)]
     (link* res data options)))
  ([data]
   (link data nil))
  ([]
   (CUlinkState_st.)))

(defn link-complete [^CUlinkState_st link-state]
  (let-release [cubin-image (byte-pointer nil)]
    (with-release [size-out (size-t-pointer 1)]
      (with-check
        (cudart/cuLinkComplete link-state cubin-image size-out)
        (capacity! cubin-image (get-entry size-out 0))))))

(defn load!
  "Load a module's data from a [[ptx]] string, `nvrtcProgram`, java path, or a binary `data`,
  for already existing module.

  See [cuModuleGetFunction](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html)
  "
  [m data]
  (module-load* (safe (pointer data)) m)
  m)

(defn module
  "Creates a new CUDA module and loads a string, nvrtc program, or binary `data`."
  ([]
   (CUmod_st.))
  ([data]
   (load! (module) data))
  ([data options]
   (load! (module) data options)))

(defrecord GridDim [^long grid-x ^long grid-y ^long grid-z ^long block-x ^long block-y ^long block-z])

(defn grid-1d
  "Creates a 1-dimensional [[GridDim]] record with grid and block dimensions x.
  Note: dim-x is the total number of threads globally, not the number of blocks."
  ([^long dim-x]
   (let [block-x (min dim-x 1024)]
     (grid-1d dim-x block-x)))
  ([^long dim-x ^long block-x]
   (let [block-x (min dim-x block-x)]
     (GridDim. (count-groups block-x dim-x) 1 1 block-x 1 1))))

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
     (GridDim. (count-groups block-x dim-x) (count-groups block-y dim-y) 1 block-x block-y 1))))

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
     (GridDim. (count-groups block-x dim-x) (count-groups block-y dim-y)
               (count-groups block-z dim-z) block-x block-y block-z))))

(defn global
  "Returns CUDA global linear memory named `name` from module `m`, with optionally specified size.

  See [cuModuleGetFunction](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html)
  "
  [^CUmod_st m ^String name]
  (let-release [dptr (long-pointer 1)]
    (with-release [byte-size (size-t-pointer 1)]
      (with-check
        (cudart/cuModuleGetGlobal ^LongPointer dptr ^SizeTPointer byte-size m name)
        {:name name}
        (->CUDevicePtr dptr (get-entry byte-size 0) false)))))

(defn set-parameter!
  "Sets the `i`th parameter in a parameter array `arr`"
  [^PointerPointer pp ^long i parameter]
  (if (< -1 i (element-count pp))
    (put-entry! pp i (pointer parameter))
    (throw (ex-info "Index out of bounds." {:requested i :available (element-count pp)})))
  pp)

(defn set-parameters!
  "Sets the `i`th parameter in a parameter array `pp` and the rest of `parameters` in places after `i`."
  [^PointerPointer pp i parameter & parameters]
  (reduce (fn [^long i param]
            (set-parameter! pp i param)
            (inc i))
          i
          (cons parameter parameters))
  pp)

(defn parameters
  "Creates an `PointerPointer`s to CUDA `params`. `params` can be any object on
  device ([[CULinearMemory]] for example), or host (arrays, numbers) that makes sense as a kernel
  parameter per CUDA specification. Use the result as an parameter argument in [[launch!]].
  "
  ([parameter & parameters]
   (let [len (if parameters (inc (count parameters)) 1)
         pp (pointer-pointer len)]
     (apply set-parameters! pp 0 parameter parameters))))

;; ====================== Execution Control ==================================

(defn function
  "Returns CUDA kernel function named `name` located in module `m`.

  See [cuModuleGetFunction](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html)
  "
  [^CUmod_st m ^String name]
  (let [res (CUfunc_st.)]
    (with-check (cudart/cuModuleGetFunction res m name) {:name name} res)))

(defn launch!
  "Invokes the kernel `fun` on a grid-dim grid of blocks, using parameters `params`.

  Optionally, you can specify the amount of shared memory that will be available to each thread block,
  and `hstream` to use for execution.

  See [cuModuleGetFunction](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html)
  "
  ([^CUfunc_st fun ^GridDim grid-dim shared-mem-bytes ^CUstream_st hstream ^PointerPointer params]
   (with-check
     (cudart/cuLaunchKernel fun (.grid-x grid-dim) (.grid-y grid-dim) (.grid-z grid-dim)
                            (.block-x grid-dim) (.block-y grid-dim) (.block-z grid-dim)
                            (int shared-mem-bytes) hstream params nil)
     {:kernel (info fun) :grid-dim grid-dim :hstream (info hstream)}
     hstream))
  ([^CUfunc_st fun ^GridDim grid-dim hstream params]
   (launch! fun grid-dim 0 hstream params))
  ([^CUfunc_st fun ^GridDim grid-dim params]
   (launch! fun grid-dim 0 nil params)))

;; ================== Stream Management ======================================

(defn stream
  "Create a stream using an optional integer `priority` and a keyword `flag`.

  Valid `flag`s are `:default` and `:non-blocking`.

  See [cuStreamCreate](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html)
  "
  ([]
   (stream* cudart/CU_STREAM_DEFAULT))
  ([flag]
   (stream* (or (stream-flags flag)
                (throw (ex-info "Invalid stream flag." {:flag flag :available stream-flags})))))
  ([^long priority flag]
   (stream* priority (or (stream-flags flag)
                         (throw (ex-info  "Invaling stream flag."
                                          {:flag flag :available stream-flags}))))))

(def default-stream
  ^{:const true
    :doc "The default per-thread stream"}
  (wrap cudart/CU_STREAM_PER_THREAD))

(defn ready?
  "Determines status (ready or not) of a compute stream or event `obj`.

  See [cuStreamQuery](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html),
  and [cuEventQuery](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html)
  "
  [obj]
  (= cudart/CUDA_SUCCESS (ready* (extract obj))))

(defn synchronize!
  "Block for the current context's or `stream`'s tasks to complete."
  ([]
   (with-check (cudart/cuCtxSynchronize) true))
  ([^CUstream_st hstream]
   (with-check (cudart/cuStreamSynchronize hstream) hstream)))

(defn add-host-fn!
  "Adds a [[host-fn]] to a compute stream, with optional `data` related to the call.
  If `data` is not provided, places `hstream` under data.

  See [cuStreamAddCallback](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html)"
  ([hstream f data]
   (add-host-fn* hstream f data)
   hstream)
  ([hstream f]
   (add-host-fn* hstream f hstream)
   hstream))

(defn listen!
  "Adds a [[host-fn]] to a compute stream, with optional `data` related to the call.
  If `data` is not provided, places `hstream` under data.

  See [cuStreamAddCallback](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html)"
  ([hstream ch data]
   (let [data (safe (pointer data))]
     (add-host-fn* hstream (host-fn* data ch) data)
     hstream))
  ([hstream ch]
   (add-host-fn* hstream (host-fn* hstream ch) hstream)
   hstream))

(defn wait-event!
  "Makes a compute stream `hstream` wait on an event `ev`

  See [cuStreamWaitEvent](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html)"
  [^CUstream_st hstream ^CUevent_st ev]
  (with-check (cudart/cuStreamWaitEvent hstream ev 0) hstream))

(defn attach-mem!
  "Attach memory `mem` of size `size`, specified by `flag` to a `hstream` asynchronously.

  Valid flags are: `:global`, `:host` and `:single` (the default).

  If :global flag is specified, the memory can be accessed by any stream on any device.
  If :host flag is specified, the program makes a guarantee that it won't access the memory on the device from any stream on a device that has no `concurrent-managed-access` capability.
  If :single flag is specified and `hStream` is associated with a device that has no `concurrent-managed-access` capability, the program makes a guarantee that it will only access the memory on the device from `hStream`. It is illegal to attach singly to the nil stream, because the nil stream is a virtual global stream and not a specific stream. An error will be returned in this case.

  When memory is associated with a single stream, the Unified Memory system will allow CPU access to this memory
  region so long as all operations in hStream have completed, regardless of whether other streams are active.
  In effect, this constrains exclusive ownership of the managed memory region by an active GPU to per-stream
  activity instead of whole-GPU activity.

  See [cuStreamAttachMemAsync](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html)."
  ([^CUstream_st hstream mem size flag]
   (let [hstream' (cond hstream (extract hstream)
                        (and (= :global flag) (null? hstream)) nil
                        :else (throw (ex-info "nil stream is a virtual global stream and not a specific stream that may be only used with :global mem-attach flag."
                                              {:flag flag :available mem-attach-flags})))]
     (attach-mem* hstream' (extract mem) size
                  (or (mem-attach-flags flag)
                      (throw (ex-info "Unknown mem-attach flag."
                                      {:flag flag :available mem-attach-flags})))))
   hstream)
  ([mem size flag]
   (attach-mem! default-stream mem size flag)))

;; ================== Event Management =======================================

(defn event
  "Creates an event specified by keyword `flags`.

  Available flags are `:default`, `:blocking-sync`, `:disable-timing`, and `:interprocess`.

  See [cuEventCreate](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html)
  "
  ([]
   (event* cudart/CU_EVENT_DEFAULT))
  ([flag & flags]
   (event* (if flags
             (mask event-flags (cons flag flags))
             (or (event-flags flag)
                 (throw (ex-info  "Unknown event flag." {:flag flag :available event-flags})))))))

(defn elapsed-time!
  "Computes the elapsed time in milliseconds between `start-event` and `end-event`.

  See [cuEventElapsedTime](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html)
  "
  ^double [^CUevent_st start-event ^CUevent_st end-event]
  (let [res (float-array 1)]
    (with-check (cudart/cuEventElapsedTime res start-event end-event) (aget res 0))))

(defn record!
  "Records an even! `ev` on optional `stream`.

  See [cuEventRecord](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html)
  "
  ([^CUstream_st stream ^CUevent_st event]
   (with-check (cudart/cuEventRecord event stream) stream))
  ([^CUevent_st event]
   (with-check (cudart/cuEventRecord event nil) default-stream)))

;; ================== Peer Context Memory Access =============================

(defn can-access-peer
  "Queries if a device may directly access a peer device's memory.

  See [cuDeviceCanAccessPeer](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PEER__ACCESS.html)
  "
  [dev peer]
  (can-access-peer* (extract dev) (extract peer)))

(defn p2p-attribute
  "Queries attributes of the link between two devices.

  See [cuDeviceGetP2PAttribute](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PEER__ACCESS.html)
  "
  [dev peer attribute]
  (p2p-attribute* (extract dev) (extract peer) (or (p2p-attributes attribute)
                               (throw (ex-info "Unknown p2p attribute"
                                               {:attribute attribute :available p2p-attributes})))))

(defn disable-peer-access!
  "Disables direct access to memory allocations in a peer context and unregisters
  any registered allocations.

  See [cuCtxDisablePeerAccess](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PEER__ACCESS.html)
  "
  ([ctx]
   (with-check (cudart/cuCtxDisablePeerAccess ctx) ctx))
  ([]
   (disable-peer-access! (current-context))))

(defn enable-peer-access!
  "Enables direct access to memory allocations in a peer context and unregisters
  any registered allocations.

  See [cuCtxEnablePeerAccess](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PEER__ACCESS.html)
  "
  ([ctx]
   (with-check (cudart/cuCtxEnablePeerAccess ctx 0) ctx)
   ctx)
  ([]
   (enable-peer-access! (current-context))))

;; ====================== Nvrtc program JIT ========================================

(defn program
  "Creates a CUDA program from the `source-code`, with an optional `name` and an optional
  hash map of `headers` (as strings) and their names."
  ([^String name ^String source-code headers]
   (program* (string-pointer name) (string-pointer source-code)
             (pointer-pointer (into-array String (vals headers)))
             (pointer-pointer (into-array String (keys headers)))))
  ([source-code headers]
   (program nil source-code headers))
  ([source-code]
   (program nil source-code nil)))

(defn program-log
  "Returns the log string generated by the previous compilation of `prog`."
  [prog]
  (program-log* prog))

(defn compile!
  "Compiles the given `prog` using a list of string `options`."
  ([prog options]
   (compile* prog (pointer-pointer (into-array String options)))
   prog)
  ([prog]
   (compile! prog nil)))

(defn ptx
  "Returns the PTX generated by the previous compilation of `prog`."
  [prog]
  (ptx* prog))
