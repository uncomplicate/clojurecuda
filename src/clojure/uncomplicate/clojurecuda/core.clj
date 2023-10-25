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
  be provided as strings (that may be stored and read from files) or binaries, written in CUDA C/C++.

  Many examples are available in ClojureCUDA [core test](https://github.com/uncomplicate/clojurecuda/blob/master/test/clojure/uncomplicate/clojurecuda/core_test.clj).
  You can see how to write CUDA [kernels here](https://github.com/uncomplicate/clojurecuda/tree/master/test/cuda/examples)
  and [here](https://github.com/uncomplicate/clojurecuda/tree/master/test/cuda/uncomplicate/clojurecuda/kernels)
  and examples of [how to load them here](https://github.com/uncomplicate/clojurecuda/tree/master/test/clojure/uncomplicate/clojurecuda/examples/).

  For more advanced examples, please read the source code of the CUDA engine of [Neanderthal linear algebra library](https://github.com/uncomplicate/neanderthal) (mainly general CUDA and cuBLAS are used there),
  and the [Deep Diamond tensor and linear algebra library](https://github.com/uncomplicate/neanderthal) (for extensive use of cuDNN).

  Here's a categorized map of core functions. Most functions throw `ExceptionInfo` in case of errors
  thrown by the CUDA driver.

  - Device management: [[init]], [[device-count]], [[device]].
  - Context management: [[context]], [[current-context]], [[current-context!]], [[put-context!]],
  [[push-context!]], [[in-context]], [[with-context]], [[with-default]].
  - Memory management: [[memcpy!]], [[mumcpy-to-host!]], [[memcpy-to-device!]], [[memset!]].
  [[mem-sub-region]], [[mem-alloc-driver]], [[mem-alloc-runtime]], [[cuda-malloc]], [[cuda-free!]]
  [[mem-alloc-pinned]], [[mem-register-pinned!]], [[mem-alloc-mapped]],
  - Module management: [[link]], [[link-complete!]], [[load!]], [[module]].
  - Execution control: [[gdid-1d]], [[grid-2d]], [[grid-3d]], [[global]], [[set-parameter!]],
  [[parameters]], [[function]], [[launch!]].
  - Stream management: [[stream]], [[default-stream]], [[ready?]], [[synchronize!]],
  [[add-host-fn!]], [[listen!]], [[wait-event!]], [[attach-mem!]].
  - Event management: [[event]], [[elapsed-time!]], [[record!]], [[can-access-peer]],
  [[p2p-attribute]], [[disable-peer-access!]], [[enable-peer-access!]].
  - NVRTC program JIT: [[program]], [[program-log]], [[compile!]], [[ptx]].

  Please see [CUDA Driver API](https://docs.nvidia.com/cuda/pdf/CUDA_Driver_API.pdf) for details
  not discussed in ClojureCUDA documentation.
  "
  (:require [uncomplicate.commons
             [core :refer [with-release let-release info bytesize sizeof size]]
             [utils :refer [mask count-groups dragan-says-ex]]]
            [uncomplicate.fluokitten.protocols :refer [extract]]
            [uncomplicate.clojure-cpp
             :refer [null? pointer byte-pointer string-pointer int-pointer long-pointer
                     size-t-pointer pointer-pointer get-entry put-entry! safe type-pointer position!
                     capacity! address]]
            [uncomplicate.clojurecuda.info :as cuda-info]
            [uncomplicate.clojurecuda.internal
             [constants :refer [ctx-flags event-flags mem-attach-flags mem-host-alloc-flags
                                mem-host-register-flags p2p-attributes stream-flags]]
             [impl :refer [->CUDevice ->CUDevicePtr add-host-fn* attach-mem* can-access-peer*
                           compile* context* cu-address* current-context* event* host-fn* link*
                           malloc-runtime* mem-alloc-host* mem-alloc-managed* mem-host-alloc*
                           mem-host-register* memcpy* memcpy-host* memset* module-load* offset
                           p2p-attribute* program* program-log* ptx* ready* set-parameter* stream*]]
             [utils :refer [with-check]]])
  (:import [org.bytedeco.javacpp Pointer LongPointer SizeTPointer PointerPointer]
           org.bytedeco.cuda.global.cudart
           [org.bytedeco.cuda.cudart CUctx_st CUlinkState_st CUmod_st CUfunc_st CUstream_st CUevent_st]))

(defn init
  "Initializes the CUDA driver. This function must be called before any other function
  from ClojureCUDA in the current process.
  See [CUDA Initialization](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__INITIALIZE.html)
  "
  []
  (with-check (cudart/cuInit 0) true))

;; ================== Device Management ====================================

(defn device-count
  "Returns the number of CUDA devices on the system.
  See [CUDA Device Management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html).
  "
  ^long []
  (let [res (int-pointer 1)]
    (with-check (cudart/cuDeviceGetCount res) (get-entry res 0))))

(defn device
  "Returns a device specified with its ordinal number `id` or string PCI Bus `id`.
  See [CUDA Device Management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html).
  "
  ([id]
   (with-release [res (int-pointer 1)]
     (with-check
       (if (number? id)
         (cudart/cuDeviceGet res (long id))
         (cudart/cuDeviceGetByPCIBusId res ^String id))
       {:device-id id}
       (->CUDevice (get-entry res 0)))))
  ([]
   (device 0)))

;; =================== Context Management ==================================

(defn context
  "Creates a CUDA context on the `device` using a keyword `flag`.
  For available flags, see [[internal.constants/ctx-flags]]. The default is none.
  The context must be released after use.

  See [CUDA Context Management](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html).
  "
  ([dev flag]
   (context* (extract dev)
             (or (ctx-flags flag)
                 (throw (ex-info "Unknown context flag." {:flag flag :available ctx-flags})))))
  ([dev]
   (context* (extract dev) 0)))

(defn current-context
  "Returns the CUDA context bound to the calling CPU thread.
  See [CUDA Context Management](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html).
  "
  []
  (current-context*))

(defn current-context!
  "Binds the specified CUDA context `ctx` to the calling CPU thread.
  See [CUDA Context Management](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html).
  "
  [ctx]
  (current-context* ctx)
  ctx)

(defn pop-context!
  "Pops the current CUDA context `ctx` from the current CPU thread.
  See [CUDA Context Management](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html).
  "
  []
  (let [ctx (CUctx_st.)]
    (with-check (cudart/cuCtxPopCurrent ctx) ctx)))

(defn push-context!
  "Pushes a context `ctx` on the current CPU thread.
  See [CUDA Context Management](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html).
  "
  [^CUctx_st ctx]
  (with-check (cudart/cuCtxPushCurrent ctx) ctx))

(defmacro in-context
  "Pushes the context `ctx` to the top of the context stack, evaluates the body with `ctx`
  as the current context, and pops the context from the stack.
  Does NOT release the context, unlike [[with-context]].
  See [CUDA Context Management](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html).
  "
  [ctx & body]
  `(try
     (push-context! ~ctx)
     ~@body
     (finally (pop-context!))))

(defmacro with-context
  "Pushes the context `ctx` to the top of the context stack, evaluates the body, and pops the context
  from the stack. Releases the context, unlike [[in-context]].
  See [CUDA Context Management](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html).
  "
  [ctx & body]
  `(with-release [ctx# ~ctx]
     (in-context ctx# ~@body)))

(defmacro with-default
  "Initializes CUDA, creates the default context and executes the body in it.
  See [CUDA Context Management](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html).
  "
  [& body]
  `(do
     (init)
     (with-release [dev# (device)]
       (with-context (context dev#)
         ~@body))))

;; ================== Memory Management  ==============================================

(defn ^:private check-size [ptr ^long offset ^long byte-count]
  (when-not (<= 0 offset (+ offset byte-count) (bytesize ptr))
    (dragan-says-ex "Requested bytes are out of the bounds of this device pointer."
                    {:offset offset :requested byte-count :available (bytesize ptr)})))

(defn memcpy!
  "Copies `byte-count` or maximum available device memory from `src` to `dst`.
  TODO mapped, pinned
  If `hstream` is provided, executes asynchronously.
  See [CUDA Memory Management](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html)
  "
  ([src dst]
   (memcpy! src dst (min (bytesize src) (bytesize dst)) nil))
  ([src dst byte-count-or-stream]
   (if (number? byte-count-or-stream)
     (do (check-size src 0 byte-count-or-stream)
         (check-size dst 0 byte-count-or-stream)
         (memcpy* dst src byte-count-or-stream nil))
     (memcpy! src dst (min (bytesize src) (bytesize dst)) byte-count-or-stream))
   dst)
  ([src dst ^long byte-count hstream]
   (check-size src 0 byte-count)
   (check-size dst 0 byte-count)
   (memcpy* dst src byte-count hstream)
   dst))

(defn memcpy-to-host!
  "Copies `byte-count` or maximum available memory from device `src` to host `dst`. Useful when `src`
  or `dst` is a generic pointer for which it cannot be determined whether it manages memory on host
  or on device (see [[cuda-malloc!]]).
  If `hstream` is provided, executes asynchronously.
  See [CUDA Memory Management](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html)
  "
  ([^Pointer src ^Pointer dst ^long byte-count hstream]
   (check-size src 0 byte-count)
   (check-size dst 0 byte-count)
   (with-check
     (if hstream
       (cudart/cuMemcpyDtoHAsync (extract dst) (address (extract src)) byte-count hstream)
       (cudart/cuMemcpyDtoH (extract dst) (address (extract src)) byte-count))
     dst))
  ([src dst count-or-stream]
   (if (integer? count-or-stream)
     (memcpy-to-host! src dst count-or-stream nil)
     (memcpy-to-host! src dst (min (bytesize src) (bytesize dst)) count-or-stream))
   dst)
  ([src dst]
   (memcpy-to-host! src dst (min (bytesize src) (bytesize dst)))
   dst))

(defn memcpy-to-device!
  "Copies `byte-count` or all possible memory from host `src` to device `dst`. Useful when `src` or
  `dst` is a generic pointer for which it cannot be determined whether it manages memory on host or
  on device (see [[cuda-malloc!]]).
  If `hstream` is provided, executes asynchronously.
  See [CUDA Memory Management](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html)
  "
  ([^Pointer src ^Pointer dst ^long byte-count hstream]
   (check-size src 0 byte-count)
   (check-size dst 0 byte-count)
   (with-check
     (if hstream
       (cudart/cuMemcpyHtoDAsync (address (extract dst)) (extract src) byte-count hstream)
       (cudart/cuMemcpyHtoD (address (extract dst)) (extract src) byte-count))
     dst))
  ([src dst count-or-stream]
   (if (integer? count-or-stream)
     (memcpy-to-device! src dst count-or-stream nil)
     (memcpy-to-device! src dst (min (bytesize src) (bytesize dst)) count-or-stream))
   dst)
  ([src dst]
   (memcpy-to-device! src dst (min (bytesize src) (bytesize dst)))
   dst))

(defn memcpy-host!
  "Copies `byte-count` or all possible memory from `src` to `dst`, one of which
  has to be accessible from the host. If `hstream` is provided, executes asynchronously.
  A polymorphic function that figures out what needs to be done. Supports everything
  except pointers created by [[cuda-malloc!]].
  See [CUDA Memory Management](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html)
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
     (memcpy-host! src dst count-or-stream nil)
     (memcpy-host* dst src (min (bytesize src) (bytesize dst)) count-or-stream))
   dst)
  ([src dst]
   (memcpy-host* dst src (min (bytesize src) (bytesize dst)))
   dst))

(defn memset!
  "Sets `n` elements or all segments of `dptr` memory to `value` (supports all Java primitive number
  types except `double`, and `long` with value larger than `Integer/MAX_VALUE`). If `hstream` is
  provided, executes asynchronously.
  See [CUDA Memory Management](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html)
  "
  ([dptr value]
   (memset* value (cu-address* dptr) (quot (bytesize dptr) (sizeof value)))
   dptr)
  ([dptr value n-or-hstream]
   (if (integer? n-or-hstream)
     (do (check-size dptr 0 (* (sizeof value) (long n-or-hstream)))
         (memset* value (cu-address* dptr) n-or-hstream))
     (memset* value (cu-address* dptr) (quot (bytesize dptr) (sizeof value)) n-or-hstream))
   dptr)
  ([dptr value ^long n hstream]
   (if hstream
     (do (check-size dptr 0 (* (sizeof value) n))
         (memset* value (cu-address* dptr) n hstream))
     (memset! dptr value n))
   dptr))

;; ==================== Driver-managed device memory ===============================================

(defn mem-sub-region
  "Creates CUDA device memory object that references a sub-region of `mem` from `origin`
  to `byte-count`, or maximum available byte size.
  "
  ([mem ^long origin ^long byte-count]
   (check-size mem origin byte-count)
   (let-release [sub-dptr (long-pointer 1)]
     (->CUDevicePtr (put-entry! sub-dptr 0 (offset mem origin)) byte-count false)))
  ([mem ^long origin]
   (mem-sub-region mem origin (bytesize mem))))

(defn mem-alloc-driver
  "Allocates the `byte-size` bytes of uninitialized memory that will be automatically managed by the
  Unified Memory system, specified by a keyword `flag`. For available flags, see [[internal.constants/mem-attach-flags]].
  Returns a CUDA device memory object, which can NOT be extracted as a `Pointer`, but can be accessed
  directly through its address in the device memory.
  See [CUDA Driver API Memory Management](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html)
  "
  ([^long byte-size flag]
   (mem-alloc-managed* (max 0 byte-size)
                       (or (mem-attach-flags flag)
                           (throw (ex-info "Unknown mem-attach flag."
                                           {:flag flag :available mem-attach-flags})))))
  ([^long byte-size]
   (mem-alloc-managed* byte-size cudart/CU_MEM_ATTACH_GLOBAL)))

;; =================== Runtime API Memory ================================================

(defn mem-alloc-runtime
  "Allocates the `byte-size` bytes of uninitialized memory that will be automatically managed by the
  Unified Memory system. Returns a CUDA device memory object managed by the CUDA runtime API, which
  can be extracted as a `Pointer`. Equivalent unwrapped `Pointer` can be created by [[cuda-malloc]].
  See [CUDA Runtime API Memory Management](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html)
  "
  ([^long byte-size type] ;;TODO functions that receive type should accept size instead of bytesize
   (if-let [t (type-pointer type)]
     (malloc-runtime* (max 0 byte-size) t)
     (throw (ex-info (format "Unknown data type: %s." (str type)) {}))))
  ([^long byte-size]
   (malloc-runtime* (max 0 byte-size))))

(defn cuda-malloc
  "Returns a `Pointer` to `byte-size` bytes of uninitialized memory that will be automatically
  managed by the Unified Memory system. The pointer is managed by the CUDA runtime API.
  Optionally, accepts a `type` of the pointer as a keyword (`:float` or `Float/TYPE` for
  `FloatPointer`, etc.).
  This pointer has to be manually released by [[cuda-free!]]. For a more seamless experience,
  use the wrapper provided by the [[mem-alloc-runtime]] function.
  See [CUDA Runtime API Memory Management](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html)
  "
  ([^long byte-size]
   (let [byte-size (max 0 byte-size)]
     (let-release [p (byte-pointer nil)]
       (with-check (cudart/cudaMalloc p byte-size) (capacity! p byte-size)))))
  ([^long byte-size type]
   (if-let [pt (type-pointer type)]
     (let [byte-size (max 0 byte-size)]
       (let-release [p (byte-pointer nil)]
         (with-check (cudart/cudaMalloc p byte-size) (pt (capacity! p byte-size)))))
     (throw (ex-info (format "Unknown data type: %s." (str type)) {})))))

(defn cuda-free!
  "Frees the runtime device memory that has been created by [[cuda-malloc]].
  See [CUDA Runtime API Memory Management](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html)
  "
  [^Pointer dptr]
  (when-not (null? dptr)
    (with-check (cudart/cudaFree (position! dptr 0))
      (do (.deallocate dptr) (.setNull dptr))))
  dptr)

;; =================== Pinned and Mapped Memory ================================================

(defn mem-alloc-pinned
  "Allocates `byte-size` bytes of uninitialized page-locked memory, 'pinned' on the host, using
  keyword `flags`. For available flags, see [[internal.constants/mem-host-alloc-flags]]; the default
  is `:none`. Optionally, accepts a `type` of the pointer as a keyword (`:float` or `Float/TYPE` for
  `FloatPointer`, etc.).
  Pinned memory is optimized for the [[memcpy-host!]] function, while 'mapped' memory is optimized
  for [[memcpy!]].
  See [CUDA Device Driver API Memory Management](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html)
  "
  ([^long byte-size]
   (mem-host-alloc* (max 0 byte-size) 0))
  ([^long byte-size type-or-flags]
   (if-let [t (type-pointer type-or-flags)]
     (mem-host-alloc* (max 0 byte-size) 0 t)
     (mem-host-alloc* (max 0 byte-size)
                      (if (keyword? type-or-flags)
                        (or (mem-host-alloc-flags type-or-flags)
                            (throw (ex-info "Unknown mem-host-alloc flag."
                                            {:flag type-or-flags :available mem-host-alloc-flags})))
                        (mask mem-host-alloc-flags type-or-flags)))))
  ([^long byte-size type flags]
   (if-let [t (type-pointer type)]
     (mem-host-alloc* (max 0 byte-size)
                      (if (keyword? flags)
                        (or (mem-host-alloc-flags flags)
                            (throw (ex-info "Unknown mem-host-alloc flag."
                                            {:flag flags :available mem-host-alloc-flags})))
                        (mask mem-host-alloc-flags flags))
                      t)
     (throw (ex-info (format "Unknown data type: %s." (str type)) {})))))

(defn mem-register-pinned!
  "Registers previously instantiated host pointer, 'pinned' from the device, using
  keyword `flags`. For available flags, see [[internal.constants/mem-host-register-flags]]; the
  default is `:none`. Returns the pinned object equivalent to the one created by [[mem-alloc-pinned]].
  Pinned memory is optimized for the [[memcpy-host!]] function, while 'mapped' memory is
  optimized for [[memcpy!]].
  See [CUDA Device Driver API Memory Management](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html)
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
  "Allocates `byte-size` bytes of uninitialized host memory, 'mapped' to the device. Optionally,
  accepts a `type` of the pointer as a keyword (`:float` or `Float/TYPE` for `FloatPointer`, etc.).
  Mapped memory is optimized for the [[memcpy!]] operation, while 'pinned' memory is optimized for
  [[memcpy-host!]].
  See [CUDA Driver API Memory Management](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html)
  "
  ([^long byte-size]
   (mem-alloc-host* (max 0 byte-size)))
  ([^long byte-size type]
   (mem-alloc-host* (max 0 byte-size) (type-pointer type))))

;; ================== Module Management =====================================

(defn link
  "Invokes the CUDA linker on data provided as a vector `[[type source <options> <name>], ...]`.
  Produces a cubin compiled for a particular Nvidia architecture.
  Please see relevant examples from the test folder.
  See [CUDA Module Management](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html)
  "
  ([data options]
   (let-release [res (CUlinkState_st.)]
     (link* res data options)))
  ([data]
   (link data nil))
  ([]
   (CUlinkState_st.)))

(defn link-complete!
  "Completes the link state created by [[link]], so that it can be loaded by the [[module]] function.
  Please see relevant examples from the test folder."
  [^CUlinkState_st link-state]
  (let-release [cubin-image (byte-pointer nil)]
    (with-release [size-out (size-t-pointer 1)]
      (with-check
        (cudart/cuLinkComplete link-state cubin-image size-out)
        (capacity! cubin-image (get-entry size-out 0))))))

(defn load!
  "Load module's data from a [[ptx]] string, nvrtc program, java path, or binary `data`.
  Please see relevant examples from the test folder.
  See [CUDA Module Management](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html)
  "
  [m data]
  (module-load* (safe (pointer data)) m)
  m)

(defn module
  "Creates a new CUDA module and loads a string, nvrtc program, or binary `data`.
  See [CUDA Module Management](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html)"
  ([]
   (CUmod_st.))
  ([data]
   (load! (module) data)))

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
  "Returns CUDA global device memory object named `name` from module `m`. Global memory is
  typically defined in C++ source files of CUDA kernels.
  See [CUDA Module Management](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html)
  "
  [^CUmod_st m ^String name]
  (let-release [dptr (long-pointer 1)]
    (with-release [byte-size (size-t-pointer 1)]
      (with-check
        (cudart/cuModuleGetGlobal ^LongPointer dptr ^SizeTPointer byte-size m name)
        {:name name}
        (->CUDevicePtr dptr (get-entry byte-size 0) false)))))

(defn set-parameter!
  "Sets the `i`th parameter in a parameter array `pp` and the rest of `parameters` in places after `i`."
  [^PointerPointer pp i parameter & parameters]
  (if (< -1 (long i) (size pp))
    (set-parameter* parameter (extract pp) i)
    (throw (ex-info "Index out of bounds." {:requested i :available (size pp)})))
  (if parameters
    (recur pp (inc (long i)) (first parameters) (next parameters))
    pp))

(defn parameters
  "Creates an `PointerPointer`s to CUDA `parameter`'s. `parameter` can be any object on
  device (Device API memory, Runtime API memory, JavaCPP pointers), or host (arrays, numbers, JavaCPP
  pointers) that makes sense as a kernel parameter per CUDA specification. Use the result as a parameter
  argument in [[launch!]].
  "
  ([parameter & parameters]
   (let-release [len (if parameters (inc (count parameters)) 1)
                 pp (pointer-pointer len)]
     (apply set-parameter! pp 0 parameter parameters))))

;; ====================== Execution Control ==================================

(defn function
  "Returns CUDA kernel function named `name` located in module `m`.
  See [CUDA Module Management](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html)
  "
  [^CUmod_st m ^String name]
  (let [res (CUfunc_st.)]
    (with-check (cudart/cuModuleGetFunction res m name) {:name name} res)))

(defn launch!
  "Invokes the kernel `fun` on a `grid-dim` grid of blocks, usinng `params` `PointerPointer`.
  Optionally, you can specify the amount of shared memory that will be available to each thread block,
  and `hstream` to use for execution.
  See [CUDA Module Management](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html)
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
  "Creates a stream using an optional integer `priority` and a keyword `flag`.
  For available flags, see [[internal.constants/stream-flags]]
  See [CUDA Stream Management](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html)
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
    :doc "The default per-thread stream."}
   cudart/CU_STREAM_PER_THREAD)

(defn ready?
  "Determines status (ready or not) of a compute stream or event `obj`.
  See [CUDA Stream Management](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html)
  and [CUDA Event Management](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html)
  "
  [obj]
  (= cudart/CUDA_SUCCESS (ready* (extract obj))))

(defn synchronize!
  "Blocks the current thread until the context's or `hstream`'s tasks complete."
  ([]
   (with-check (cudart/cuCtxSynchronize) true))
  ([^CUstream_st hstream]
   (with-check (cudart/cuStreamSynchronize hstream) hstream)))

(defn add-host-fn!
  "Adds host function `f` to a compute stream, with optional `data` related to the call.
  If `data` is not provided, places `hstream` under data.
  "
  ([hstream f data]
   (add-host-fn* hstream f data)
   hstream)
  ([hstream f]
   (add-host-fn* hstream f hstream)
   hstream))

(defn listen!
  "Adds a host function listener to a compute stream, with optional `data` related to the call,
  and connects it to a Clojure channel `chan`. If `data` is not provided, places `hstream` under data.
  "
  ([hstream ch data]
   (let [data (safe (pointer data))]
     (add-host-fn* hstream (host-fn* data ch) data)
     hstream))
  ([hstream ch]
   (add-host-fn* hstream (host-fn* hstream ch) hstream)
   hstream))

(defn wait-event!
  "Makes a compute stream `hstream` wait on an event `ev`.
  See [CUDA Event Management](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html)
  "
  [^CUstream_st hstream ^CUevent_st ev]
  (with-check (cudart/cuStreamWaitEvent hstream ev 0) hstream))

(defn attach-mem!
  "Attaches memory `mem` of size `size`, specified by `flag` to a `hstream` asynchronously.
  For available flags, see [[internal.constants/mem-attach-flags]]. Te default is `:single`.
  If :global flag is specified, the memory can be accessed by any stream on any device.
  If :host flag is specified, the program makes a guarantee that it won't access the memory on
  the device from any stream on a device that has no `concurrent-managed-access` capability.
  If :single flag is specified and `hStream` is associated with a device that has no
  `concurrent-managed-access` capability, the program makes a guarantee that it will only access
  the memory on the device from `hStream`. It is illegal to attach singly to the nil stream,
  because the nil stream is a virtual global stream and not a specific stream. An error will
  be returned in this case.

  When memory is associated with a single stream, the Unified Memory system will allow CPU access
  to this memory region so long as all operations in hStream have completed, regardless of whether
  other streams are active. In effect, this constrains exclusive ownership of the managed memory
  region by an active GPU to per-stream activity instead of whole-GPU activity.

  See [CUDA Stream Management](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html)."
  ([^CUstream_st hstream mem ^long byte-size flag]
   (attach-mem* (or (extract hstream)
                    (when-not (= :global flag)
                      (throw (ex-info "nil stream is a virtual global stream and not a specific stream that may be only used with :global mem-attach flag."
                                      {:flag flag :available mem-attach-flags}))))
                (cu-address* mem) byte-size
                (or (mem-attach-flags flag)
                    (throw (ex-info "Unknown mem-attach flag."
                                    {:flag flag :available mem-attach-flags}))))
   hstream)
  ([mem byte-size flag]
   (attach-mem! default-stream mem byte-size flag)))

;; ================== Event Management =======================================

(defn event
  "Creates an event specified by keyword `flags`. For available flags, see
  [[internal.constants/event-flags]].
  See [CUDA Event Management](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html)
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
  See [CUDA Event Management](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html)
  "
  ^double [^CUevent_st start-event ^CUevent_st end-event]
  (let [res (float-array 1)]
    (with-check (cudart/cuEventElapsedTime res start-event end-event) (aget res 0))))

(defn record!
  "Records an even! `ev` on optional `stream`.
  See [CUDA Event Management](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html)
  "
  ([^CUstream_st stream ^CUevent_st event]
   (with-check (cudart/cuEventRecord event stream) stream))
  ([^CUevent_st event]
   (with-check (cudart/cuEventRecord event nil) default-stream)))

;; ================== Peer Context Memory Access =============================

(defn can-access-peer
  "Queries if a device may directly access a peer device's memory.
  See [CUDA Peer Access Management](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PEER__ACCESS.html)
  "
  [dev peer]
  (can-access-peer* (extract dev) (extract peer)))

(defn p2p-attribute
  "Queries attributes of the link between two devices.
  See [CUDA Peer Access Management](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PEER__ACCESS.html)
  "
  [dev peer attribute]
  (p2p-attribute* (extract dev) (extract peer) (or (p2p-attributes attribute)
                                                   (throw (ex-info "Unknown p2p attribute."
                                                                   {:attribute attribute :available p2p-attributes})))))

(defn disable-peer-access!
  "Disables direct access to memory allocations in a peer context and unregisters any registered allocations.
  See [CUDA Peer Access Management](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PEER__ACCESS.html)
  "
  ([ctx]
   (with-check (cudart/cuCtxDisablePeerAccess ctx) ctx))
  ([]
   (disable-peer-access! (current-context))))

(defn enable-peer-access!
  "Enables direct access to memory allocations in a peer context and unregisters any registered allocations.
  See [CUDA Peer Access Management](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PEER__ACCESS.html)
  "
  ([ctx]
   (with-check (cudart/cuCtxEnablePeerAccess ctx 0) ctx)
   ctx)
  ([]
   (enable-peer-access! (current-context))))

;; ====================== Nvrtc program JIT ========================================

(defn program
  "Creates a CUDA program from the `source-code`, with an optional `name` and an optional
  hash map of `headers` (as strings) and their names.
  "
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
