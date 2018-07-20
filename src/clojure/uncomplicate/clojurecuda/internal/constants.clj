;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.clojurecuda.internal.constants
  "Defines constants and mappings from/to CUDA constants."
  (:import [jcuda.driver CUctx_flags JCudaDriver CUdevice_attribute CUcomputemode CUmemAttach_flags
            CUfunc_cache CUdevice_P2PAttribute CUlimit CUsharedconfig CUstream_flags CUevent_flags
            CUjit_option CUjitInputType]))

;; ==================== Keyword mapping ======================================

(def ^{:const true
       :doc "Available context flags defined in the CUDA standard."}
  ctx-flags
  {:sched-auto CUctx_flags/CU_CTX_SCHED_AUTO
   :sched-spin CUctx_flags/CU_CTX_SCHED_SPIN
   :sched-yield CUctx_flags/CU_CTX_SCHED_YIELD
   :sched-blocking-sync CUctx_flags/CU_CTX_SCHED_BLOCKING_SYNC
   :map-host CUctx_flags/CU_CTX_MAP_HOST
   :lmem-resize-to-max CUctx_flags/CU_CTX_LMEM_RESIZE_TO_MAX})

(def ^{:const true
       :doc "Available context limits."}
  ctx-limits
  {:stack-size CUlimit/CU_LIMIT_STACK_SIZE
   :malloc-heap-size CUlimit/CU_LIMIT_MALLOC_HEAP_SIZE
   :printf-fifo-size CUlimit/CU_LIMIT_PRINTF_FIFO_SIZE
   :dev-runtime-sync-depth CUlimit/CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH
   :dev-runtime-pending-launch-count CUlimit/CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT})

(def ^{:const true
       :doc "Available shared memory configurations."}
  shared-config-map
  {:default-bank-size CUsharedconfig/CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE
   :four-byte-bank-size CUsharedconfig/CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE
   :eight-byte-bank-size CUsharedconfig/CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE})

(defn dec-shared-config [^long config]
  (case config
    0 :default-bank-size
    1 :four-byte-bank-size
    2 :eight-byte-bank-size
    config))

(def ^{:const true
       :doc "Available device P2P attributes."}
  p2p-attributes
  {:performance-rank CUdevice_P2PAttribute/CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK
   :access-supported CUdevice_P2PAttribute/CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED
   :native-atomic-supported CUdevice_P2PAttribute/CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED})

(defn dec-compute-mode [^long mode]
  (case mode
    0 :default
    1 :exclusive
    2 :prohibited
    3 :exclusive-process
    mode) )

(def ^{:const true
       :doc "Available flags for the [[core/mem-host-alloc]] function."}
  mem-host-alloc-flags
  {:portable JCudaDriver/CU_MEMHOSTALLOC_PORTABLE
   :devicemap JCudaDriver/CU_MEMHOSTALLOC_DEVICEMAP
   :writecombined JCudaDriver/CU_MEMHOSTALLOC_WRITECOMBINED})

(def ^{:const true
       :doc "Available flags for the [[core/mem-host-register]] function."}
  mem-host-register-flags
  {:portable JCudaDriver/CU_MEMHOSTREGISTER_PORTABLE
   :devicemap JCudaDriver/CU_MEMHOSTREGISTER_DEVICEMAP})

(def ^{:const true
       :doc "Available flags for the [[core/mem-host-attach]] function."}
  mem-attach-flags
  {:global CUmemAttach_flags/CU_MEM_ATTACH_GLOBAL
   :host CUmemAttach_flags/CU_MEM_ATTACH_HOST})

(def ^{:const true
       :doc "Available flags for the [[core/mem-host-attach]] function."}
  stream-flags
  {:default CUstream_flags/CU_STREAM_DEFAULT
   :non-blocking CUstream_flags/CU_STREAM_NON_BLOCKING})

(defn dec-stream-flag [^long flag]
  (case flag
    0 :default
    1 :non-blocking
    flag))

(def ^{:const true
       :doc "Available flags for the [[core/event]] function."}
  event-flags
  {:default CUevent_flags/CU_EVENT_DEFAULT
   :blocking-sync CUevent_flags/CU_EVENT_BLOCKING_SYNC
   :disable-timing CUevent_flags/CU_EVENT_DISABLE_TIMING
   :interprocess CUevent_flags/CU_EVENT_INTERPROCESS})

(def ^{:const true
       :doc "Available config for the [[core/cache-config!]] function."}
  func-cache-config
  {:prefer-none CUfunc_cache/CU_FUNC_CACHE_PREFER_NONE
   :prefer-shared CUfunc_cache/CU_FUNC_CACHE_PREFER_SHARED
   :prefer-L1 CUfunc_cache/CU_FUNC_CACHE_PREFER_L1
   :prefer-equal CUfunc_cache/CU_FUNC_CACHE_PREFER_EQUAL})

(defn dec-func-cache-config [^long mode]
  (case mode
    3 :prefer-equal
    2 :prefer-L1
    0 :prefer-none
    1 :prefer-shared
    mode))

(def ^{:const true
       :doc "Available jit options defined in the CUDA standard."}
  jit-options
  {:max-registers CUjit_option/CU_JIT_MAX_REGISTERS
   :threads-per-block CUjit_option/CU_JIT_THREADS_PER_BLOCK
   :wall-time CUjit_option/CU_JIT_WALL_TIME
   :info-log-buffer CUjit_option/CU_JIT_INFO_LOG_BUFFER
   :info-log-buffer-size-bytes CUjit_option/CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES
   :error-log-buffer CUjit_option/CU_JIT_ERROR_LOG_BUFFER
   :error-log-buffer-size-bytes CUjit_option/CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES
   :optimization-level CUjit_option/CU_JIT_OPTIMIZATION_LEVEL
   :target-from-cucontext CUjit_option/CU_JIT_TARGET_FROM_CUCONTEXT
   :target CUjit_option/CU_JIT_TARGET
   :fallback-strategy CUjit_option/CU_JIT_FALLBACK_STRATEGY
   :generate-debug-info CUjit_option/CU_JIT_GENERATE_DEBUG_INFO
   :log-verbose CUjit_option/CU_JIT_LOG_VERBOSE
   :generate-line-info CUjit_option/CU_JIT_GENERATE_LINE_INFO
   :cache-mode CUjit_option/CU_JIT_CACHE_MODE})

(def ^{:const true
       :doc "Available jit input types defined in the CUDA standard."}
  jit-input-types
  {:cubin CUjitInputType/CU_JIT_INPUT_CUBIN
   :ptx CUjitInputType/CU_JIT_INPUT_PTX
   :object CUjitInputType/CU_JIT_INPUT_OBJECT
   :library CUjitInputType/CU_JIT_INPUT_LIBRARY})
