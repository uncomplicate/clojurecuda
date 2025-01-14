;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.clojurecuda.internal.impl
  (:require [uncomplicate.commons
             [core :refer [with-release let-release Releaseable release info Bytes bytesize Entries
                           size* size]]
             [utils :as cu :refer [dragan-says-ex]]]
            [uncomplicate.fluokitten.protocols :refer [Comonad extract]]
            [uncomplicate.clojure-cpp :as cpp
             :refer [put-entry! pointer safe int-pointer pointer-pointer byte-pointer size-t-pointer
                     get-entry get-string null? long-pointer PointerCreator TypedPointerCreator
                     clong-pointer short-pointer char-pointer double-pointer float-pointer pointer-seq
                     capacity! address Accessor get! put! get-keyword]]
            [uncomplicate.clojurecuda.internal
             [constants :refer [cu-result-codes jit-input-types jit-options nvrtc-result-codes]]
             [utils :refer [with-check]]]
            [clojure.core.async :refer [go >!]])
  (:import java.nio.file.Path
           java.io.File
           [clojure.lang IFn AFn Seqable]
           [org.bytedeco.javacpp Pointer BytePointer PointerPointer LongPointer IntPointer]
           [org.bytedeco.cuda.global cudart nvrtc]
           [org.bytedeco.cuda.cudart CUctx_st CUstream_st CUevent_st CUmod_st CUlinkState_st]
           org.bytedeco.cuda.nvrtc._nvrtcProgram
           [uncomplicate.clojure_cpp StringPointer KeywordPointer]
           [uncomplicate.clojurecuda.internal.javacpp CUHostFn CUStreamCallback]))

(defprotocol CUPointer
  (cu-address* [this])
  (device? [this]))

(defprotocol Parameter
  (set-parameter* [this pp i]))

(extend-type Object
  Parameter
  (set-parameter* [parameter pp i]
    (put-entry! pp i (pointer parameter))))

;; ==================== Release resources =======================

(deftype CUDevice [^int dev]
  Object
  (hashCode [_]
    dev)
  (equals [_ y]
    (and (instance? CUDevice y) (= dev (.dev ^CUDevice y))))
  (toString [_]
    (format "#Device[:cuda, %d]" dev))
  Comonad
  (extract [_]
    dev))

(extend-type CUctx_st
  Releaseable
  (release [this]
    (locking this
      (cudart/cuCtxDestroy this)
      (.deallocate this)
      (.setNull this)
      true)))

(extend-type CUstream_st
  Releaseable
  (release [this]
    (locking this
      (cudart/cuStreamDestroy this)
      (.deallocate this)
      (.setNull this)
      true)))

(extend-type CUevent_st
  Releaseable
  (release [this]
    (locking this
      (cudart/cuEventDestroy this)
      (.deallocate this)
      (.setNull this)
      true)))

(extend-type CUmod_st
  Releaseable
  (release [this]
    (locking this
      (cudart/cuModuleUnload this)
      (.deallocate this)
      (.setNull this)
      true)))

(extend-type CUlinkState_st
  Releaseable
  (release [this]
    (locking this
      (cudart/cuLinkDestroy this)
      (.deallocate this)
      (.setNull this)
      true)))

(extend-type _nvrtcProgram
  Releaseable
  (release [this]
    (locking this
      (nvrtc/nvrtcDestroyProgram this)
      (.deallocate this)
      (.setNull this)
      true)))

;; ================== Module Management =====================================

(defprotocol ModuleLoad
  (module-load* [data m])
  (link-add* [data link-state type opts vals]))

(defn enc-jit-options [options]
  (map (fn [[option value]]
         [(or (jit-options option)
              (throw (ex-info "Unknown jit option." {:option option :available jit-options})))
          (safe (pointer value))])
       options))

(defn check-options [^IntPointer options ^Pointer option-values]
  (when-not (= (size options) (size option-values))
    (throw (ex-info "Inconsistent number of options provided."
                    {:requested (size options) :provided (size option-values)}))))

(defn link-add-data* [^CUlinkState_st link-state type ^Pointer data ^String name
                      ^IntPointer options ^Pointer option-values]
  (let [type (int (or (jit-input-types type)
                      (throw (ex-info "Invalid jit input type."
                                      {:type type :available jit-input-types}))))]
    (check-options options option-values)
    (with-check (cudart/cuLinkAddData link-state type data (bytesize data) name
                                      (size options) options option-values)
      {:data data} link-state)))

(defn link-add-file* [^CUlinkState_st link-state type ^String file-name
                      ^IntPointer options ^Pointer option-values]
  (let [type (int (or (jit-input-types type)
                      (throw (ex-info "Invalid jit input type."
                                      {:type type :available jit-input-types}))))]
    (check-options options option-values)
    (with-check (cudart/cuLinkAddFile link-state type file-name
                                      (size options) options option-values)
      {:file file-name} link-state)))

(defn link*
  [^CUlinkState_st link-state data options]
  (let [[opts vals] (enc-jit-options options)]
    (let-release [opts (int-pointer opts)
                  vals (pointer-pointer vals)]
      (with-check (cudart/cuLinkCreate (size opts) opts ^PointerPointer vals link-state)
        (doseq [[type d options name] data]
          (if name
            (link-add-data* link-state type d name opts vals)
            (link-add* d link-state type opts vals))))))
  link-state)

(extend-type String
  ModuleLoad
  (module-load* [data m]
    (with-check (cudart/cuModuleLoadData ^CUmod_st m (byte-pointer data)) {:data data} m))
  (link-add* [data link-state type opts vals]
    (link-add-data* link-state type (byte-pointer data) "unnamed" opts vals)))

(extend-type Pointer
  ModuleLoad
  (module-load* [data m]
    (with-check (cudart/cuModuleLoadData m data)
      {:data data} m))
  (link-add* [data link-state type opts vals]
    (link-add-data* link-state type data "unnamed" opts vals)))

(extend-type Path
  ModuleLoad
  (module-load* [file-path m]
    (let [file-name (.toString file-path)]
      (with-check (cudart/cuModuleLoad ^CUmod_st m (str file-name)) {:file (str file-path)} m)))
  (link-add* [file-path link-state type opts vals]
    (link-add-file* link-state type (.toString file-path) opts vals)))

(extend-type File
  ModuleLoad
  (module-load* [file m]
    (with-check (cudart/cuModuleLoad ^CUmod_st m (str file)) {:file (str file)} m))
  (link-add* [file link-state type opts vals]
    (link-add-file* link-state type (.toString file) opts vals)))

;; ====================== Nvrtc program JIT ========================================

(defn ^:private nvrtc-error
  "Converts an CUDA Nvrtc error code to an ExceptionInfo with richer, user-friendly information."
  ([^long err-code details]
   (let [err (get nvrtc-result-codes err-code err-code)]
     (ex-info (format "NVRTC error: %s." err)
              {:name err :code err-code :type :nvrtc-error :details details})))
  ([err-code]
   (nvrtc-error err-code nil)))

(defmacro ^:private with-check-nvrtc
  "Evaluates `form` if `err-code` is not zero (`:success`), otherwise throws
  an appropriate `ExceptionInfo` with decoded informative details.
  It helps with CUDA nvrtc methods that return error codes directly, while
  returning computation results through mutating arguments.
  "
  ([err-code form]
   `(cu/with-check nvrtc-error ~err-code ~form)))

(defn program*
  [^BytePointer name ^BytePointer source-code
   ^PointerPointer source-headers ^PointerPointer include-names]
  (let-release [res (_nvrtcProgram.)]
    (with-check-nvrtc
      (nvrtc/nvrtcCreateProgram res source-code name
                                (size source-headers) source-headers include-names)
      res)))

(defn program-log*
  "Returns the log string generated by the previous compilation of `program`."
  [^_nvrtcProgram program]
  (with-release [log-size (size-t-pointer 1)]
    (with-check-nvrtc (nvrtc/nvrtcGetProgramLogSize program log-size)
      (with-release [log (byte-pointer (get-entry log-size 0))]
        (with-check-nvrtc (nvrtc/nvrtcGetProgramLog program log) (get-string log))))))

(defn compile*
  "Compiles the given `program` using an array of string `options`."
  ([^_nvrtcProgram program ^PointerPointer options]
   (let [err (nvrtc/nvrtcCompileProgram program (size options) options)]
     (if (= 0 err)
       program
       (throw (nvrtc-error err (program-log* program)))))))

(defn ptx*
  "Returns the PTX generated by the previous compilation of `program`."
  [^_nvrtcProgram program]
  (with-release [ptx-size (size-t-pointer 1)]
    (with-check-nvrtc (nvrtc/nvrtcGetPTXSize program ptx-size)
      (let-release [ptx (byte-pointer (get-entry ptx-size 0))]
        (with-check-nvrtc (nvrtc/nvrtcGetPTX program ptx)
          ptx)))))

(extend-type _nvrtcProgram
  ModuleLoad
  (module-load* [program m]
    (with-check (cudart/cuModuleLoadData ^CUmod_st m (ptx* program)) m))
  (link-add* [program link-state type opts vals]
    (link-add-data* link-state type (ptx* program) "unnamed" opts vals)))

;; =================== Context Management ==================================

(defn context*
  "Creates a CUDA context on the `device` using a raw integer `flag`.
  For available flags, see [[constants/ctx-flags]].
  "
  [^long dev ^long flags]
  (let [res (CUctx_st.)]
    (with-check (cudart/cuCtxCreate res flags dev)
      {:dev (info dev) :flags flags}
      res)))

(defn current-context*
  "If `ctx` is provided, bounds it as current. Returns the CUDA context bound to the calling CPU thread."
  ([]
   (let [ctx (CUctx_st.)]
     (with-check (cudart/cuCtxGetCurrent ctx) ctx)))
  ([^CUctx_st ctx]
   (with-check (cudart/cuCtxSetCurrent ctx) ctx)))

;; ==================== Linear memory ================================================

(defprotocol MemSet
  (memset* [this dptr n] [this dptr n hstream]))

(extend-type Byte
  MemSet
  (memset*
    ([this dptr n]
     (with-check (cudart/cuMemsetD8 dptr this n) dptr))
    ([this dptr n hstream]
     (with-check (cudart/cuMemsetD8Async dptr this n hstream) dptr))))

(extend-type Short
  MemSet
  (memset*
    ([this dptr n]
     (with-check (cudart/cuMemsetD16 dptr this n) dptr))
    ([this dptr n hstream]
     (with-check (cudart/cuMemsetD16Async dptr this n hstream) dptr))))

(extend-type Integer
  MemSet
  (memset*
    ([this dptr n]
     (with-check (cudart/cuMemsetD32 dptr this n) dptr))
    ([this dptr n hstream]
     (with-check (cudart/cuMemsetD32Async dptr this n hstream) dptr))))

(extend-type Float
  MemSet
  (memset*
    ([this dptr n]
     (with-check (cudart/cuMemsetD32 dptr (Float/floatToIntBits this) n) dptr))
    ([this dptr n hstream]
     (with-check (cudart/cuMemsetD32Async dptr (Float/floatToIntBits this) n hstream) dptr))))

(extend-type Double
  MemSet
  (memset*
    ([this dptr n]
     (if (= 0.0 this)
       (with-check (cudart/cuMemsetD32 dptr (int 0) (* 2 (long n))) dptr)
       (dragan-says-ex "Only zeroes are suported in double memset! function." {:value this})))
    ([this dptr n hstream]
     (if (= 0.0 this)
       (with-check (cudart/cuMemsetD32Async dptr (int 0) (* 2 (long n)) hstream) dptr)
       (dragan-says-ex "Only zeroes are suported in double memset! function." {:value this})))))

(extend-type Long
  MemSet
  (memset*
    ([this dptr n]
     (if (= 0 this)
       (with-check (cudart/cuMemsetD32 dptr (int 0) (* 2 (long n))) dptr)
       (dragan-says-ex "Only zeroes are suported in long memset! function." {:value this})))
    ([this dptr n hstream]
     (if (= 0 this)
       (with-check (cudart/cuMemsetD32Async dptr (int 0) (* 2 (long n)) hstream) dptr)
       (dragan-says-ex "Only zeroes are suported in long memset! function." {:value this})))))

(defprotocol Memcpy
  "An object that represents memory that participates in CUDA operations.
  It can be on the device, or on the host. Built-in implementations:
  CUDA pointers, JavaCPP pointers, Java primitive arrays, etc.
  "
  (memcpy-host* [dst src size] [dst src size hstream])
  (memcpy* [dst src size] [dst src size hstream]))

(defn offset ^long [dptr ^long offset]
  (if (<= 0 offset (bytesize dptr))
    (+ (long (cu-address* dptr)) offset)
    (dragan-says-ex "Requested bytes are out of the bounds of this device pointer."
                    {:offset offset :size (bytesize dptr)})))

(deftype CUDevicePtr [^LongPointer daddr ^long byte-size master]
  Object
  (hashCode [_]
    (hash-combine (hash daddr) byte-size))
  (equals [_ y]
    (and (instance? CUDevicePtr y) (= (get-entry daddr 0) (cu-address* y))))
  (toString [_]
    (format "#DevicePtr[:cuda, 0x%x, %d bytes]" (get-entry daddr 0) byte-size))
  Releaseable
  (release [_]
    (locking daddr
      (when-not (null? daddr)
        (when master
          (with-check (cudart/cuMemFree (get-entry daddr 0)) true))
        (release daddr))
      true))
  Comonad
  (extract [_]
    (extract daddr))
  CUPointer
  (cu-address* [_]
    (get-entry daddr 0))
  (device? [_]
    true)
  Bytes
  (bytesize* [_]
    byte-size)
  Entries
  (size* [_]
    byte-size)
  (sizeof* [_]
    Byte/BYTES)
  Parameter
  (set-parameter* [_ pp i]
    (put-entry! pp i daddr))
  Memcpy
  (memcpy-host* [this src byte-count]
    (with-check
      (cudart/cuMemcpyHtoD (get-entry daddr 0) (safe (pointer src)) byte-count)
      this))
  (memcpy-host* [this src byte-count hstream]
    (with-check
      (cudart/cuMemcpyHtoDAsync (get-entry daddr 0) (safe (pointer src)) byte-count hstream)
      this))
  (memcpy* [this src byte-count]
    (with-check
      (cudart/cuMemcpy (get-entry daddr 0) (cu-address* src) byte-count)
      this))
  (memcpy* [this src byte-count hstream]
    (with-check
      (cudart/cuMemcpyAsync (get-entry daddr 0) (cu-address* src) byte-count hstream)
      this)))

(defn mem-alloc-managed*
  ([^long size ^long flag]
   (let-release [daddr (long-pointer 1)]
     (with-check (cudart/cuMemAllocManaged daddr size flag)
       (->CUDevicePtr daddr size true)))))

;; =================== Runtime Memory ===============================================

(defn cupointer-memcpy*
  ([dst src ^long byte-count]
   (with-check
     (if (instance? Pointer src)
       (cudart/cudaMemcpy (safe (pointer dst)) (extract src) byte-count cudart/cudaMemcpyDefault)
       (cudart/cuMemcpy (cu-address* dst) (cu-address* src) byte-count))
     dst))
  ([dst src ^long byte-count hstream]
   (with-check
     (if (instance? Pointer src)
       (cudart/cudaMemcpyAsync (safe (pointer dst)) (extract src)
                               byte-count cudart/cudaMemcpyDefault hstream)
       (cudart/cuMemcpyAsync (cu-address* dst) (cu-address* src) byte-count hstream))
     dst)))

(defn offset-address [^Pointer p]
  (+ (.address (safe p)) (* (.sizeof p) (.position p))))

(deftype CURuntimePtr [^Pointer dptr master]
  Object
  (hashCode [_]
    (hash dptr))
  (equals [_ y]
    (and (instance? CURuntimePtr y) (= dptr (.-dptr ^CURuntimePtr y) 0)))
  (toString [this]
    (format "#RuntimePtr[:cuda, 0x%x, %d bytes]" (cu-address* this) (bytesize dptr)))
  Releaseable
  (release [_]
    (locking dptr
      (when-not (null? dptr)
        (when master
          (with-check (cudart/cudaFree (.position dptr 0)) (.setNull dptr))))
      true))
  Comonad
  (extract [_]
    (offset-address dptr))
  CUPointer
  (cu-address* [_]
    (offset-address dptr))
  (device? [_]
    true)
  PointerCreator
  (pointer* [_]
    dptr)
  (pointer* [_ i]
    (pointer dptr i))
  TypedPointerCreator
  (byte-pointer [_]
    (byte-pointer dptr))
  (clong-pointer [_]
    (clong-pointer dptr))
  (size-t-pointer [_]
    (clong-pointer dptr))
  (pointer-pointer [_]
    (pointer-pointer dptr))
  (char-pointer [_]
    (char-pointer dptr))
  (short-pointer [_]
    (short-pointer dptr))
  (int-pointer [_]
    (int-pointer dptr))
  (long-pointer [_]
    (long-pointer dptr))
  (float-pointer [_]
    (float-pointer dptr))
  (double-pointer [_]
    (double-pointer dptr))
  Bytes
  (bytesize* [_]
    (bytesize dptr))
  Entries
  (size* [_]
    (size* dptr))
  (sizeof* [_]
    (.sizeof dptr))
  Seqable
  (seq [_]
    (pointer-seq dptr))
  Parameter
  (set-parameter* [this pp i]
    (put-entry! pp i (pointer (offset-address dptr))))
  Memcpy
  (memcpy-host* [this src byte-count]
    (with-check
      (cudart/cuMemcpyHtoD (offset-address dptr) (safe (pointer src)) byte-count)
      this))
  (memcpy-host* [this src byte-count hstream]
    (with-check
      (cudart/cuMemcpyHtoDAsync (offset-address dptr) (safe (pointer src)) byte-count hstream)
      this))
  (memcpy* [this src byte-count]
    (cupointer-memcpy* this src byte-count))
  (memcpy* [this src byte-count hstream]
    (cupointer-memcpy* this src byte-count hstream)))

(defn malloc-runtime*
  ([^long size]
   (let-release [p (byte-pointer nil)]
     (with-check (cudart/cudaMalloc p size)
       (->CURuntimePtr (capacity! p size) true))))
  ([^long size pointer-type]
   (let-release [p (byte-pointer nil)]
     (with-check (cudart/cudaMalloc p size)
       (->CURuntimePtr (pointer-type (capacity! p size)) true)))))

;; =================== Pinned Memory ================================================

(defn free-pinned [hp]
  (with-check (cudart/cuMemFreeHost hp) (release hp)))

(defn unregister-pinned [hp]
  (with-check (cudart/cuMemHostUnregister hp) hp))

(deftype CUPinnedPtr [^Pointer hptr master release-fn]
  Object
  (hashCode [_]
    (hash hptr))
  (equals [this y]
    (and (instance? CUPinnedPtr y) (= (offset-address hptr) (cu-address* y))))
  (toString [this]
    (format "#PinnedPtr[:cuda, 0x%x, %d bytes]" (offset-address hptr) (bytesize hptr)))
  Releaseable
  (release [_]
    (locking hptr
      (when-not (null? hptr)
        (when master
          (release-fn (.position hptr 0))))
      true))
  Comonad
  (extract [_]
    (extract hptr))
  CUPointer
  (cu-address* [_]
    (offset-address hptr))
  (device? [_]
    false)
  PointerCreator
  (pointer* [_]
    hptr)
  (pointer* [_ i]
    (pointer hptr i))
  TypedPointerCreator
  (byte-pointer [_]
    (byte-pointer hptr))
  (clong-pointer [_]
    (clong-pointer hptr))
  (size-t-pointer [_]
    (clong-pointer hptr))
  (pointer-pointer [_]
    (pointer-pointer hptr))
  (char-pointer [_]
    (char-pointer hptr))
  (short-pointer [_]
    (short-pointer hptr))
  (int-pointer [_]
    (int-pointer hptr))
  (long-pointer [_]
    (long-pointer hptr))
  (float-pointer [_]
    (float-pointer hptr))
  (double-pointer [_]
    (double-pointer hptr))
  Bytes
  (bytesize* [_]
    (bytesize hptr))
  Entries
  (size* [_]
    (size* hptr))
  (sizeof* [_]
    (.sizeof hptr))
  Seqable
  (seq [_]
    (pointer-seq hptr))
  Accessor
  (get-entry [_]
    (get-entry hptr))
  (get-entry [_ i]
    (get-entry hptr i))
  (put-entry! [this value]
    (put-entry! hptr value)
    this)
  (put-entry! [this i value]
    (put-entry! hptr i value)
    this)
  (get! [_ arr]
    (get! hptr arr)
    arr)
  (get! [_ arr offset length]
    (get! hptr arr offset length)
    arr)
  (put! [this obj]
    (put! hptr obj)
    this)
  (put! [_ obj offset length]
    (put! hptr obj offset length))
  Parameter
  (set-parameter* [_ pp i]
    (put-entry! pp i (pointer (offset-address hptr))))
  Memcpy
  (memcpy-host* [this src byte-count]
    (with-check (cudart/cuMemcpyDtoH hptr (offset-address hptr) byte-count) this))
  (memcpy-host* [this src byte-count hstream]
    (with-check (cudart/cuMemcpyDtoHAsync hptr (offset-address hptr) byte-count hstream) this))
  (memcpy* [this src byte-count]
    (cupointer-memcpy* this src byte-count))
  (memcpy* [this src byte-count hstream]
    (cupointer-memcpy* this src byte-count hstream)))

(defn mem-host-alloc*
  "Allocates `byte-size` bytes of page-locked memory, 'pinned' on the host, using raw integer `flags`.
  For available flags, see [[constants/mem-host-alloc-flags]]. The memory is not initialized.
  `byte-size` must be greater than `0`.
  See [cuMemHostAlloc](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html).
  "
  ([^long byte-size ^long flags]
   (let-release [p (byte-pointer nil)]
     (with-check (cudart/cuMemHostAlloc p byte-size flags)
       (->CUPinnedPtr (capacity! p byte-size) true free-pinned))))
  ([^long byte-size ^long flags pointer-type]
   (let-release [p (byte-pointer nil)]
     (with-check (cudart/cuMemHostAlloc p byte-size flags)
       (->CUPinnedPtr (pointer-type (capacity! p byte-size)) true free-pinned)))))

(defn mem-host-register*
  "Registers previously allocated host `Pointer` and pins it, using raw integer `flags`.
  See [cuMemHostRegister](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html).
  "
  ([hptr ^long flags]
   (with-check (cudart/cuMemHostRegister hptr (bytesize hptr) flags)
     (->CUPinnedPtr hptr true unregister-pinned)))
  ([hptr ^long flags pointer-type]
   (with-check (cudart/cuMemHostRegister hptr (bytesize hptr) flags)
     (let [tp (pointer-type (capacity! hptr size))]
       (->CUPinnedPtr tp true unregister-pinned)))))

(deftype CUMappedPtr [^Pointer hptr master]
  Object
  (hashCode [_]
    (hash hptr))
  (equals [this y]
    (and (instance? CUMappedPtr y) (= (cu-address* this) (cu-address* y))))
  (toString [this]
    (format "#PinnedPtr[:cuda, 0x%x, %d bytes]" (cu-address* this) (bytesize hptr)))
  Releaseable
  (release [_]
    (locking hptr
      (when-not (null? hptr)
        (when master
          (with-check (cudart/cuMemFreeHost (.position hptr 0))
            (release hptr))))
      true))
  Comonad
  (extract [_]
    (offset-address hptr))
  CUPointer
  (cu-address* [_]
    (offset-address hptr))
  (device? [_]
    false)
  PointerCreator
  (pointer* [_]
    hptr)
  (pointer* [_ i]
    (pointer hptr i))
  TypedPointerCreator
  (byte-pointer [_]
    (byte-pointer hptr))
  (clong-pointer [_]
    (clong-pointer hptr))
  (size-t-pointer [_]
    (clong-pointer hptr))
  (pointer-pointer [_]
    (pointer-pointer hptr))
  (char-pointer [_]
    (char-pointer hptr))
  (short-pointer [_]
    (short-pointer hptr))
  (int-pointer [_]
    (int-pointer hptr))
  (long-pointer [_]
    (long-pointer hptr))
  (float-pointer [_]
    (float-pointer hptr))
  (double-pointer [_]
    (double-pointer hptr))
  Bytes
  (bytesize* [_]
    (bytesize hptr))
  Entries
  (size* [_]
    (size* hptr))
  (sizeof* [_]
    (.sizeof hptr))
  Seqable
  (seq [_]
    (pointer-seq hptr))
  Accessor
  (get-entry [_]
    (get-entry hptr))
  (get-entry [_ i]
    (get-entry hptr i))
  (put-entry! [this value]
    (put-entry! hptr value)
    this)
  (put-entry! [this i value]
    (put-entry! hptr i value)
    this)
  (get! [_ arr]
    (get! hptr arr)
    arr)
  (get! [_ arr offset length]
    (get! hptr arr offset length)
    arr)
  (put! [this obj]
    (put! hptr obj)
    this)
  (put! [_ obj offset length]
    (put! hptr obj offset length))
  Parameter
  (set-parameter* [_ pp i]
    (put-entry! pp i (pointer (offset-address hptr))))
  Memcpy
  (memcpy-host* [this src byte-count]
    (if (device? src)
      (with-check (cudart/cuMemcpy (offset-address hptr) (cu-address* src) byte-count) this)
      (cpp/memcpy! (safe (pointer src)) (extract hptr)))
    this)
  (memcpy-host* [this src byte-count hstream]
    (with-check
      (if (device? src)
        (cudart/cuMemcpyAsync (offset-address hptr) (cu-address* src) byte-count hstream)
        (cudart/cuMemcpyHtoDAsync (offset-address hptr) (safe (pointer src)) byte-count hstream))
      this))
  (memcpy* [this src byte-count]
    (cupointer-memcpy* this src byte-count))
  (memcpy* [this src byte-count hstream]
    (cupointer-memcpy* this src byte-count hstream)))

(defn mem-alloc-host*
  "Allocates `byte-size` bytes of page-locked memory, 'mapped' to the device.
  For available flags, see [constants/mem-host-alloc-flags]
  The memory is not initialized. `byte-size` must be greater than `0`.
  See [cuMemAllocHost](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html).
  "
  ([^long byte-size]
   (let-release [p (byte-pointer nil)]
     (with-check (cudart/cuMemAllocHost p byte-size)
       (->CUMappedPtr (capacity! p byte-size) true))))
  ([^long byte-size pointer-type]
   (let-release [p (byte-pointer nil)]
     (with-check (cudart/cuMemAllocHost p byte-size)
       (->CUMappedPtr (pointer-type (capacity! p byte-size)) true)))))

;; =============== Host memory  =================================

(extend-type Pointer
  CUPointer
  (cu-address* [this]
    (offset-address this))
  (device? [_]
    false)
  Parameter
  (set-parameter* [parameter pp i]
    (put-entry! pp i (pointer (offset-address parameter))))
  Memcpy
  (memcpy-host*
    ([this src byte-count]
     (with-check
       (if (instance? Pointer src)
         (cudart/cudaMemcpy (extract this) (safe (pointer src)) cudart/cudaMemcpyDefault byte-count)
         (cudart/cuMemcpyDtoH (extract this) (cu-address* src) byte-count))
       this))
    ([this src byte-count hstream]
     (with-check
       (if (instance? Pointer src)
         (cudart/cudaMemcpyAsync (extract this) (safe (pointer src))
                                 cudart/cudaMemcpyDefault byte-count hstream)
         (cudart/cuMemcpyDtoHAsync (extract this) (cu-address* src) byte-count hstream))
       this)))
  (memcpy*
    ([this src byte-count]
     (with-check
       (if (instance? Pointer src)
         (cudart/cudaMemcpy (extract this) (safe (pointer src)) byte-count cudart/cudaMemcpyDefault)
         (cudart/cuMemcpy (offset-address (extract this)) (cu-address* src) byte-count))
       this))
    ([this src byte-count hstream]
     (with-check
       (if (instance? Pointer src)
         (cudart/cudaMemcpyAsync (extract this) (safe (pointer src))
                                 byte-count cudart/cudaMemcpyDefault hstream)
         (cudart/cuMemcpyAsync (offset-address (extract this)) (cu-address* src) byte-count hstream))
       this))))

;; ================== Stream Management ======================================

(defn stream*
  "Create a stream using an optional `priority` and an integer `flag`.
  See [cuStreamCreate](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html)
  "
  ([^long flag]
   (let [res (CUstream_st.)]
     (with-check (cudart/cuStreamCreate res flag) res)))
  ([^long priority ^long flag]
   (let [res (CUstream_st.)]
     (with-check (cudart/cuStreamCreateWithPriority res flag priority) res))))

(defn ready*
  "Determines status (ready or not) of a compute stream or event.
  See [cuStreamQuery](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html),
  and [cuEventQuery](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html)
  "
  [obj]
  (case (class obj)
    CUstream_st (cudart/cuStreamQuery obj)
    CUevent_st (cudart/cuEventQuery obj)
    cudart/CUDA_ERROR_NOT_READY))

(defrecord StreamCallbackInfo [status data])

(deftype StreamCallback [ch]
  IFn
  (invoke [_ _ status data]
    (go (>! ch (->StreamCallbackInfo (get cu-result-codes status status) (extract data)))))
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(defprotocol HostFn
  (host-fn* [type ch]))

(extend-type KeywordPointer
  HostFn
  (host-fn* [_ ch]
    (fn [data]
      (go (>! ch (get-keyword (byte-pointer data)))))))

(extend-type StringPointer
  HostFn
  (host-fn* [_ ch]
    (fn [data]
      (go (>! ch (get-string (byte-pointer data)))))))

(extend-type Pointer
  HostFn
  (host-fn* [_ ch]
    (fn [data]
      (go (>! ch data)))))

(defn add-host-fn*
  [^CUstream_st hstream ^IFn f ^Pointer data]
  (let-release [hostfn (CUHostFn. f)]
    (with-check (cudart/cuLaunchHostFunc hstream hostfn data)
      hstream)))

(defn attach-mem*
  "Attach memory of `byte-size`, specified by an integer `flag` to a `hstream` asynchronously.
  See [cuStreamAttachMemAsync](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html).
  "
  ([^CUstream_st hstream mem byte-size flag]
   (with-check (cudart/cuStreamAttachMemAsync hstream mem byte-size flag) hstream)))

;; ================== Event Management =======================================

(defn event*
  "Creates an event specified by integer `flags`.
  See [cuEventCreate](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html)
  "
  [^long flags]
  (let [res (CUevent_st.)]
    (with-check (cudart/cuEventCreate res flags) res)))

;; ================== Peer Context Memory Access =============================

(defn can-access-peer*
  "queries if a device may directly access a peer device's memory.
  see [cudevicecanaccesspeer](http://docs.nvidia.com/cuda/cuda-driver-api/group__cuda__peer__access.html)
  "
  [^long dev ^long peer]
  (with-release [res (int-pointer 1)]
    (with-check (cudart/cuDeviceCanAccessPeer ^IntPointer res dev peer)
      (pos? (int (get-entry res 0))))))

(defn p2p-attribute*
  "Queries attributes of the link between two devices.
  See [cuDeviceGetP2PAttribute](http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PEER__ACCESS.html)
  "
  [^long dev ^long peer ^long attribute]
  (let [res (int-pointer 1)]
    (with-check
      (cudart/cudaDeviceGetP2PAttribute ^IntPointer res attribute dev peer)
      (pos? (int (get-entry res 0))))))

;; ================ print-method ============================================

(defn format-pointer [title p ^java.io.Writer w]
  (.write w (format "#%s[:cuda, 0x%x]" title (address p))))

(defmethod print-method CUDevice [p ^java.io.Writer w]
  (.write w (str p)))

(defmethod print-method CUctx_st [p w]
  (format-pointer "Context" p w))

(defmethod print-method CUstream_st [p w]
  (format-pointer "Stream" p w))

(defmethod print-method CUevent_st [p w]
  (format-pointer "Event" p w))

(defmethod print-method CUmod_st [p w]
  (format-pointer "Module" p w))

(defmethod print-method CUlinkState_st [p w]
  (format-pointer "LinkState" p w))

(defmethod print-method _nvrtcProgram [p w]
  (format-pointer "Program" p w))

(defmethod print-method CUDevicePtr [p w]
  (format-pointer "DevicePtr" p w))

(defmethod print-method CUPinnedPtr [p w]
  (format-pointer "PinnedPtr" p w))

(defmethod print-method CUMappedPtr [p w]
  (format-pointer "MappedPtr" p w))
