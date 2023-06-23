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
  (:import [org.bytedeco.cuda.global cudart nvrtc]))

;; ==================== Keyword mapping ======================================

(def ^{:const true
       :doc "Available context flags defined in the CUDA standard."}
  ctx-flags
  {:blocking-sync cudart/CU_CTX_BLOCKING_SYNC
   :coredump cudart/CU_CTX_COREDUMP_ENABLE
   :flags-mask cudart/CU_CTX_FLAGS_MASK
   :lmem-resize-to-max cudart/CU_CTX_LMEM_RESIZE_TO_MAX
   :map-host cudart/CU_CTX_MAP_HOST
   :sched-auto cudart/CU_CTX_SCHED_AUTO
   :sched-blocking-sync cudart/CU_CTX_SCHED_BLOCKING_SYNC
   :sched-mask cudart/CU_CTX_SCHED_MASK
   :sched-spin cudart/CU_CTX_SCHED_SPIN
   :sched-yield cudart/CU_CTX_SCHED_YIELD
   :sync-memops cudart/CU_CTX_SYNC_MEMOPS
   :user-coredump cudart/CU_CTX_USER_COREDUMP_ENABLE})

(def ^{:const true
       :doc "Available context limits."}
  ctx-limits
  {:dev-runtime-pending-launch-count cudart/CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT
   :dev-runtime-sync-depth cudart/CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH
   :malloc-heap-size cudart/CU_LIMIT_MALLOC_HEAP_SIZE
   :max cudart/CU_LIMIT_MAX
   :max-l2-fetch-granularity cudart/CU_LIMIT_MAX_L2_FETCH_GRANULARITY
   :persisting-l2-cache-size cudart/CU_LIMIT_PERSISTING_L2_CACHE_SIZE
   :printf-fifo-size cudart/CU_LIMIT_PRINTF_FIFO_SIZE
   :stack-size cudart/CU_LIMIT_STACK_SIZE})

(def ^{:const true
       :doc "Available shared memory configurations."}
  shared-config-map
  {:default-bank-size cudart/CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE
   :four-byte-bank-size cudart/CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE
   :eight-byte-bank-size cudart/CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE})

(defn dec-shared-config [^long config]
  (case config
    0 :default-bank-size
    1 :four-byte-bank-size
    2 :eight-byte-bank-size
    config))

(def ^{:const true
       :doc "Available device P2P attributes."}
  p2p-attributes
  {:access-access-supported cudart/CU_DEVICE_P2P_ATTRIBUTE_ACCESS_ACCESS_SUPPORTED
   :access-supported cudart/CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED
   :cuda-array-access-supported cudart/CU_DEVICE_P2P_ATTRIBUTE_CUDA_ARRAY_ACCESS_SUPPORTED
   :native-atomic-supported cudart/CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED
   :performance-rank cudart/CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK})

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
  {:portable cudart/CU_MEMHOSTALLOC_PORTABLE
   :devicemap cudart/CU_MEMHOSTALLOC_DEVICEMAP
   :writecombined cudart/CU_MEMHOSTALLOC_WRITECOMBINED})

(def ^{:const true
       :doc "Available flags for the [[core/mem-host-register]] function."}
  mem-host-register-flags
  {:devicemap cudart/CU_MEMHOSTREGISTER_DEVICEMAP
   :iomemory cudart/CU_MEMHOSTREGISTER_IOMEMORY
   :portable cudart/CU_MEMHOSTREGISTER_PORTABLE
   :read-onlyp cudart/CU_MEMHOSTREGISTER_READ_ONLY})

(def ^{:const true
       :doc "Available flags for the [[core/mem-host-attach]] function."}
  mem-attach-flags
  {:global cudart/CU_MEM_ATTACH_GLOBAL
   :host cudart/CU_MEM_ATTACH_HOST
   :single cudart/CU_MEM_ATTACH_SINGLE})

(def ^{:const true
       :doc "Available flags for the [[core/mem-host-attach]] function."}
  stream-flags
  {:default cudart/CU_STREAM_DEFAULT
   :non-blocking cudart/CU_STREAM_NON_BLOCKING})

(defn dec-stream-flag [^long flag]
  (case flag
    0 :default
    1 :non-blocking
    flag))

(def ^{:const true
       :doc "Available flags for the [[core/event]] function."}
  event-flags
  {:blocking-sync cudart/CU_EVENT_BLOCKING_SYNC
   :default cudart/CU_EVENT_DEFAULT
   :disable-timing cudart/CU_EVENT_DISABLE_TIMING
   :interprocess cudart/CU_EVENT_INTERPROCESS})

(def ^{:const true
       :doc "Available config for the [[core/cache-config!]] function."}
  func-cache-config
  {:prefer-none cudart/CU_FUNC_CACHE_PREFER_NONE
   :prefer-shared cudart/CU_FUNC_CACHE_PREFER_SHARED
   :prefer-L1 cudart/CU_FUNC_CACHE_PREFER_L1
   :prefer-equal cudart/CU_FUNC_CACHE_PREFER_EQUAL})

(defn dec-func-cache-config [^long mode]
  (case mode
    0 :prefer-none
    1 :prefer-shared
    2 :prefer-L1
    3 :prefer-equal
    mode))

(def ^{:const true
       :doc "Available jit options defined in the CUDA standard."}
  jit-options
  {:cache-mode cudart/CU_JIT_CACHE_MODE
   :cache-option-ca cudart/CU_JIT_CACHE_OPTION_CA
   :cache-option-cg cudart/CU_JIT_CACHE_OPTION_CG
   :cache-option-none cudart/CU_JIT_CACHE_OPTION_NONE
   :error-log-buffer cudart/CU_JIT_ERROR_LOG_BUFFER
   :error-log-buffer-size-bytes cudart/CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES
   :fallback-strategy cudart/CU_JIT_FALLBACK_STRATEGY
   :fast-compile-strategy cudart/CU_JIT_FAST_COMPILE
   :fma cudart/CU_JIT_FMA
   :ftz cudart/CU_JIT_FTZ
   :generate-debug-info cudart/CU_JIT_GENERATE_DEBUG_INFO
   :generate-line-info cudart/CU_JIT_GENERATE_LINE_INFO
   :global-symbol-addresses cudart/CU_JIT_GLOBAL_SYMBOL_ADDRESSES
   :global-symbol-count cudart/CU_JIT_GLOBAL_SYMBOL_COUNT
   :global-symbol-names cudart/CU_JIT_GLOBAL_SYMBOL_NAMES
   :info-log-buffer cudart/CU_JIT_INFO_LOG_BUFFER
   :info-log-buffer-size-bytes cudart/CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES
   :input-cubin cudart/CU_JIT_INPUT_CUBIN
   :input-fatbinary cudart/CU_JIT_INPUT_FATBINARY
   :input-library cudart/CU_JIT_INPUT_LIBRARY
   :input-nvvm cudart/CU_JIT_INPUT_NVVM
   :input-object cudart/CU_JIT_INPUT_OBJECT
   :input-ptx cudart/CU_JIT_INPUT_PTX
   :log-verbose cudart/CU_JIT_LOG_VERBOSE
   :lto cudart/CU_JIT_LTO
   :max-registers cudart/CU_JIT_MAX_REGISTERS
   :new-sm3x-opt cudart/CU_JIT_NEW_SM3X_OPT
   :num-input-tupes cudart/CU_JIT_NUM_INPUT_TYPES
   :num-options cudart/CU_JIT_NUM_OPTIONS
   :optimization-level cudart/CU_JIT_OPTIMIZATION_LEVEL
   :optimize-unused-device-variables cudart/CU_JIT_OPTIMIZE_UNUSED_DEVICE_VARIABLES
   :position-independent-code cudart/CU_JIT_POSITION_INDEPENDENT_CODE
   :prec-div cudart/CU_JIT_PREC_DIV
   :prec-sqrt cudart/CU_JIT_PREC_SQRT
   :referenced-kernel-count cudart/CU_JIT_REFERENCED_KERNEL_COUNT
   :referenced-kernel-names cudart/CU_JIT_REFERENCED_KERNEL_NAMES
   :referenced-variable-count cudart/CU_JIT_REFERENCED_VARIABLE_COUNT
   :referenced-variable-names cudart/CU_JIT_REFERENCED_VARIABLE_NAMES
   :target cudart/CU_JIT_TARGET
   :target-from-cucontext cudart/CU_JIT_TARGET_FROM_CUCONTEXT
   :threads-per-block cudart/CU_JIT_THREADS_PER_BLOCK
   :wall-time cudart/CU_JIT_WALL_TIME})

(def ^{:const true
       :doc "Available jit input types defined in the CUDA standard."}
  jit-input-types
  {:cubin cudart/CU_JIT_INPUT_CUBIN
   :ptx cudart/CU_JIT_INPUT_PTX
   :fatbinary cudart/CU_JIT_INPUT_FATBINARY
   :object cudart/CU_JIT_INPUT_OBJECT
   :library cudart/CU_JIT_INPUT_LIBRARY
   :nvvm cudart/CU_JIT_INPUT_NVVM
   :num cudart/CU_JIT_NUM_INPUT_TYPES})

(def ^{:const true
       :doc "CUDA Error messages as defined in CUresult."}
  cu-result-codes
  {cudart/CUDA_SUCCESS :success
   cudart/CUDA_ERROR_ALREADY_ACQUIRED :already-acquired
   cudart/CUDA_ERROR_ALREADY_MAPPED :already-mapped
   cudart/CUDA_ERROR_ARRAY_IS_MAPPED :array-is-mapped
   cudart/CUDA_ERROR_ASSERT :assert
   cudart/CUDA_ERROR_CAPTURED_EVENT :captured-event
   cudart/CUDA_ERROR_CDP_NOT_SUPPORTED :cdp-not-supported
   cudart/CUDA_ERROR_CDP_VERSION_MISMATCH :sdp-version-mismatch
   cudart/CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE :compat-not-supported-on-device
   cudart/CUDA_ERROR_CONTEXT_ALREADY_CURRENT :context-already-current
   cudart/CUDA_ERROR_CONTEXT_ALREADY_IN_USE :context-already-in-use
   cudart/CUDA_ERROR_CONTEXT_IS_DESTROYED :context-is-destroyed
   cudart/CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE :cooperative-launch-too-large
   cudart/CUDA_ERROR_DEINITIALIZED :deinitialized
   cudart/CUDA_ERROR_DEVICE_NOT_LICENSED :device-not-licensed
   cudart/CUDA_ERROR_DEVICE_UNAVAILABLE :unavailable
   cudart/CUDA_ERROR_ECC_UNCORRECTABLE :ecc-uncorrectable
   cudart/CUDA_ERROR_EXTERNAL_DEVICE :external-device
   cudart/CUDA_ERROR_FILE_NOT_FOUND :file-not-found
   cudart/CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE :graph-exec-update-failure
   cudart/CUDA_ERROR_HARDWARE_STACK_ERROR :hardware-stack-errox
   cudart/CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED :host-memory-already-registered
   cudart/CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED :host-memory-not-registered
   cudart/CUDA_ERROR_ILLEGAL_ADDRESS :illegal-address
   cudart/CUDA_ERROR_ILLEGAL_INSTRUCTION :illegal-instruction
   cudart/CUDA_ERROR_ILLEGAL_STATE :illegal-state
   cudart/CUDA_ERROR_INVALID_ADDRESS_SPACE :invalid-address-space
   cudart/CUDA_ERROR_INVALID_CLUSTER_SIZE :invalid-cluster-size
   cudart/CUDA_ERROR_INVALID_CONTEXT :invalid-context
   cudart/CUDA_ERROR_INVALID_DEVICE :invalid-device
   cudart/CUDA_ERROR_INVALID_GRAPHICS_CONTEXT :invalid-graphics-context
   cudart/CUDA_ERROR_INVALID_HANDLE :invalid-handle
   cudart/CUDA_ERROR_INVALID_IMAGE :invalid-image
   cudart/CUDA_ERROR_INVALID_PC :invalid-pc
   cudart/CUDA_ERROR_INVALID_PTX :invalid-ptx
   cudart/CUDA_ERROR_INVALID_SOURCE :invalid-source
   cudart/CUDA_ERROR_INVALID_VALUE :invalid-value
   cudart/CUDA_ERROR_JIT_COMPILATION_DISABLED :jit-compilation-disabled
   cudart/CUDA_ERROR_JIT_COMPILER_NOT_FOUND :jit-compiler-not-found
   cudart/CUDA_ERROR_LAUNCH_FAILED :launch-failed
   cudart/CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING :launch-incompatible-texturing
   cudart/CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES :launch-out-of-resources
   cudart/CUDA_ERROR_LAUNCH_TIMEOUT :launch-timeout
   cudart/CUDA_ERROR_MAP_FAILED :map-failed
   cudart/CUDA_ERROR_MISALIGNED_ADDRESS :misaligned-address
   cudart/CUDA_ERROR_MPS_CLIENT_TERMINATED :client-terminated
   cudart/CUDA_ERROR_MPS_CONNECTION_FAILED :connection-failed
   cudart/CUDA_ERROR_MPS_MAX_CLIENTS_REACHED :clients-reached
   cudart/CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED :connection-reached
   cudart/CUDA_ERROR_MPS_RPC_FAILURE :rpc-failure
   cudart/CUDA_ERROR_MPS_SERVER_NOT_READY :server-not-ready
   cudart/CUDA_ERROR_NO_BINARY_FOR_GPU :binary-for-gpu
   cudart/CUDA_ERROR_NO_DEVICE :no-device
   cudart/CUDA_ERROR_NOT_FOUND :not-found
   cudart/CUDA_ERROR_NOT_INITIALIZED :not-initialized
   cudart/CUDA_ERROR_NOT_MAPPED :not-mapped
   cudart/CUDA_ERROR_NOT_MAPPED_AS_ARRAY :not-mapped-as-array
   cudart/CUDA_ERROR_NOT_MAPPED_AS_POINTER :mapped-as-pointer
   cudart/CUDA_ERROR_NOT_READY :not-ready
   cudart/CUDA_ERROR_NOT_SUPPORTED :not-supported
   cudart/CUDA_ERROR_NVLINK_UNCORRECTABLE :nvlink-uncorrectable
   cudart/CUDA_ERROR_OPERATING_SYSTEM :operating-system
   cudart/CUDA_ERROR_OUT_OF_MEMORY :out-of-memory
   cudart/CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED :already-enabled
   cudart/CUDA_ERROR_PEER_ACCESS_NOT_ENABLED :access-not-enabled
   cudart/CUDA_ERROR_PEER_ACCESS_UNSUPPORTED :access-unsupported
   cudart/CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE :context-active
   cudart/CUDA_ERROR_PROFILER_ALREADY_STARTED :profiler-already-started
   cudart/CUDA_ERROR_PROFILER_ALREADY_STOPPED :profiler-already-stopped
   cudart/CUDA_ERROR_PROFILER_DISABLED :profiler-disabled
   cudart/CUDA_ERROR_PROFILER_NOT_INITIALIZED :profiler-not-initialized
   cudart/CUDA_ERROR_SHARED_OBJECT_INIT_FAILED :shared-object-init-failed
   cudart/CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND :shared-object-symblol-not-found
   cudart/CUDA_ERROR_STREAM_CAPTURE_IMPLICIT :stream-capture-implicit
   cudart/CUDA_ERROR_STREAM_CAPTURE_INVALIDATED :stream-capture-invalidated
   cudart/CUDA_ERROR_STREAM_CAPTURE_ISOLATION :stream-capture-isolation
   cudart/CUDA_ERROR_STREAM_CAPTURE_MERGE :stream-capture-merge
   cudart/CUDA_ERROR_STREAM_CAPTURE_UNJOINED :stream-capture-unjoined
   cudart/CUDA_ERROR_STREAM_CAPTURE_UNMATCHED :stream-capture-unmatched
   cudart/CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED :stream-capture-unsupported
   cudart/CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD :stream-capture-wrong-thread
   cudart/CUDA_ERROR_STUB_LIBRARY :stub-library
   cudart/CUDA_ERROR_SYSTEM_DRIVER_MISMATCH :driver-mismatch
   cudart/CUDA_ERROR_SYSTEM_NOT_READY :system-not-ready
   cudart/CUDA_ERROR_TIMEOUT :timeout
   cudart/CUDA_ERROR_TOO_MANY_PEERS :too-many-peers
   cudart/CUDA_ERROR_UNKNOWN :unknown
   cudart/CUDA_ERROR_UNMAP_FAILED :unmap-failed
   cudart/CUDA_ERROR_UNSUPPORTED_DEVSIDE_SYNC :devside-sync
   cudart/CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY :exec-affinity
   cudart/CUDA_ERROR_UNSUPPORTED_LIMIT :unsupported-limit
   cudart/CUDA_ERROR_UNSUPPORTED_PTX_VERSION :unsupported-ptx-version})

(def ^{:const true
       :doc "CUDA Error messages as defined in nvrtc."}
  nvrtc-result-codes
  {nvrtc/NVRTC_SUCCESS :success
   nvrtc/NVRTC_ERROR_BUILTIN_OPERATION_FAILURE :builtin-operation-failure
   nvrtc/NVRTC_ERROR_COMPILATION :compilation
   nvrtc/NVRTC_ERROR_INVALID_INPUT :invalid-input
   nvrtc/NVRTC_ERROR_INTERNAL_ERROR :internal-error
   nvrtc/NVRTC_ERROR_INVALID_OPTION :invalid-option
   nvrtc/NVRTC_ERROR_INVALID_PROGRAM :invalid-program
   nvrtc/NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID :name-expression-not-valid
   nvrtc/NVRTC_ERROR_OUT_OF_MEMORY :out-of-memory
   nvrtc/NVRTC_ERROR_PROGRAM_CREATION_FAILURE :program-creation-failure
   nvrtc/NVRTC_ERROR_TIME_FILE_WRITE_FAILED :time-file-write-ahead})
