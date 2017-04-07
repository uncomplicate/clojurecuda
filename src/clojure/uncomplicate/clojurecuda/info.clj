;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.clojurecuda.info
  "Info functions for all CUDA objects (devices, etc...).
  "
  (:require [clojure.string :as str]
            [uncomplicate.fluokitten.core :refer [fmap]]
            [uncomplicate.clojurecuda
             [constants :refer :all]
             [utils :refer [with-check maybe]]])
  (:import jcuda.Pointer
           [jcuda.driver JCudaDriver CUdevice CUdevice_attribute]))

;; =================== Info* utility macros ===============================

(defmacro ^:private info-string*
  ([method object size]
   `(let [res# (byte-array ~size)
          err# (~method res# ~size ~object)]
      (with-check err#
        (String. res# 0 (int (loop [i# 0]
                               (if (or (= i# ~size) (= (byte 0x00) (aget res# i#)))
                                 i#
                                 (recur (inc i#)))))))))
  ([method object]
   `(info-string* ~method ~object 32)))

(defmacro ^:private info-attribute* [method object attribute]
  `(let [res# (int-array 1)
         err# (~method res# ~attribute ~object)]
     (long (with-check err# (aget res# 0)))))

;;========================= Protocols =====================================

(defprotocol Info
  (info [this info-type] [this]))

;; =================== Device info  =======================================

(defn device-name [^CUdevice device]
  (info-string* JCudaDriver/cuDeviceGetName device))

(defn async-engine-count ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT))

(defn can-map-host-memory [^CUdevice device]
  (pos? (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                         CUdevice_attribute/CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY)))

(defn clock-rate ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_CLOCK_RATE))

(defn compute-capability-major ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR))

(defn compute-capability-minor ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR))

(defn compute-mode [^CUdevice device]
  (dec-compute-mode (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                                     CUdevice_attribute/CU_DEVICE_ATTRIBUTE_COMPUTE_MODE)))

(defn concurrent-kernels ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS))

(defn ecc-enabled [^CUdevice device]
  (pos? (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                         CUdevice_attribute/CU_DEVICE_ATTRIBUTE_ECC_ENABLED)))

(defn global-L1-cache-supported [^CUdevice device]
  (pos? (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                         CUdevice_attribute/CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED)))

(defn global-memory-bus-width ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH))

(defn integrated [^CUdevice device]
  (pos? (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                         CUdevice_attribute/CU_DEVICE_ATTRIBUTE_INTEGRATED)))

(defn kernel-exec-timeout [^CUdevice device]
  (pos? (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                         CUdevice_attribute/CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT)))

(defn L2-cache-size ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE))

(defn local-L1-cache-supported [^CUdevice device]
  (pos? (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                         CUdevice_attribute/CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED)))

(defn managed-memory [^CUdevice device]
  (pos? (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                         CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY)))

(defn max-block-dim-x ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X))

(defn max-block-dim-y ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y))

(defn max-block-dim-z ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z))

(defn max-grid-dim-x ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X))

(defn max-grid-dim-y ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y))

(defn max-grid-dim-z ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z))

(defn max-pitch ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAX_PITCH))

(defn max-registers-per-block ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK))

(defn max-registers-per-multiprocessor ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR))

(defn max-shared-memory-per-block ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK))

(defn max-shared-memory-per-multiprocessor ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR))

(defn max-threads-per-block ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK))

(defn max-threads-per-multiprocessor ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR))

(defn maximum-surface1d-layered-layers ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS))

(defn maximum-surface1d-layered-width ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH))

(defn maximum-surface1d-width ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH))

(defn maximum-surface2d-height ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT))

(defn maximum-surface2d-width ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH))

(defn maximum-surface2d-layered-height ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT))

(defn maximum-surface2d-layered-width ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH))

(defn maximum-surface2d-layered-layers ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS))

(defn maximum-surface3d-depth ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH))

(defn maximum-surface3d-height ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT))

(defn maximum-surface3d-width ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH))

(defn maximum-surfacecubemap-layered-width ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH))

(defn maximum-surfacecubemap-layered-layers ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS))

(defn maximum-surfacecubemap-width ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH))

(defn maximum-texture1d-layered-width ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH))

(defn maximum-texture1d-layered-layers ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS))

(defn maximum-texture1d-linear-width ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH))

(defn maximum-texture1d-mipmapped-width ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH))

(defn maximum-texture1d-width ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH))

(defn maximum-texture2d-height ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT))

(defn maximum-texture2d-layered-height ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT))

(defn maximum-texture2d-layered-layers ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS))

(defn maximum-texture2d-linear-height ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT))

(defn maximum-texture2d-linear-pitch ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH))

(defn maximum-texture2d-linear-width ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH))

(defn maximum-texture2d-mipmapped-width ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH))

(defn maximum-texture2d-mipmapped-height ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT))

(defn maximum-texture2d-width ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH))

(defn maximum-texture3d-depth ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH))

(defn maximum-texture3d-depth-alternate ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE))

(defn maximum-texture3d-height ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT))

(defn maximum-texture3d-height-alternate ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE))

(defn maximum-texture3d-width ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH))

(defn maximum-texture3d-width-alternate ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE))

(defn maximum-texturecubemap-layered-layers ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS))

(defn maximum-texturecubemap-layered-width ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH))

(defn maximum-texturecubemap-width ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH))

(defn memory-clock-rate ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE))

(defn multi-gpu-board [^CUdevice device]
  (pos? (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                         CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD)))

(defn multi-gpu-board-group-id ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID))

(defn multiprocessor-count ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT))

(defn pci-bus-id ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_PCI_BUS_ID))

(defn pci-device-id ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID))

(defn pci-domain-id ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID))

(defn stream-priorities-supported [^CUdevice device]
  (pos? (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                         CUdevice_attribute/CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED)))

(defn surface-alignment ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT))

(defn tcc-driver [^CUdevice device]
  (pos? (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                         CUdevice_attribute/CU_DEVICE_ATTRIBUTE_TCC_DRIVER)))

(defn texture-alignment ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT))

(defn texture-pitch-alignment ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT))

(defn total-constant-memory ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY))

(defn unified-addressing [^CUdevice device]
  (pos? (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                         CUdevice_attribute/CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING)))

(defn warp-size ^long [^CUdevice device]
  (info-attribute* JCudaDriver/cuDeviceGetAttribute device
                   CUdevice_attribute/CU_DEVICE_ATTRIBUTE_WARP_SIZE))

(def device-attributes
  {:name device-name
   :async-engine-count async-engine-count
   :can-map-host-memory can-map-host-memory
   :clock-rate clock-rate
   :compute-capability-major compute-capability-major
   :compute-capability-minor compute-capability-minor
   :compute-mode compute-mode
   :concurrent-kernels concurrent-kernels
   :ecc-enabled ecc-enabled
   :global-L1-cache-supported global-L1-cache-supported
   :global-memory-bus-width global-memory-bus-width
   :integrated integrated
   :kernel-exec-timeout kernel-exec-timeout
   :L2-cache-size L2-cache-size
   :local-L1-cache-supported local-L1-cache-supported
   :managed-memory managed-memory
   :max-block-dim-x max-block-dim-x
   :max-block-dim-y max-block-dim-y
   :max-block-dim-z max-block-dim-z
   :max-grid-dim-x max-grid-dim-x
   :max-grid-dim-y max-grid-dim-y
   :max-grid-dim-z max-grid-dim-z
   :max-pitch max-pitch
   :max-registers-per-block max-registers-per-block
   :max-registers-per-multiprocessor max-registers-per-multiprocessor
   :max-shared-memory-per-block max-shared-memory-per-block
   :max-shared-memory-per-multiprocessor max-shared-memory-per-multiprocessor
   :max-threads-per-block max-threads-per-block
   :max-threads-per-multiprocessor max-threads-per-multiprocessor
   :maximum-surface1d-layered-layers maximum-surface1d-layered-layers
   :maximum-surface1d-layered-width maximum-surface1d-layered-width
   :maximum-surface1d-width maximum-surface1d-width
   :maximum-surface2d-height maximum-surface2d-height
   :maximum-surface2d-width maximum-surface2d-width
   :maximum-surface2d-layered-height maximum-surface2d-layered-height
   :maximum-surface2d-layered-width maximum-surface2d-layered-width
   :maximum-surface2d-layered-layers maximum-surface2d-layered-layers
   :maximum-surface3d-depth maximum-surface3d-depth
   :maximum-surface3d-height maximum-surface3d-height
   :maximum-surface3d-width maximum-surface3d-width
   :maximum-surfacecubemap-layered-width maximum-surfacecubemap-layered-width
   :maximum-surfacecubemap-layered-layers maximum-surfacecubemap-layered-layers
   :maximum-surfacecubemap-width maximum-surfacecubemap-width
   :maximum-texture1d-layered-width maximum-texture1d-layered-width
   :maximum-texture1d-layered-layers maximum-texture1d-layered-layers
   :maximum-texture1d-linear-width maximum-texture1d-linear-width
   :maximum-texture1d-mipmapped-width maximum-texture1d-mipmapped-width
   :maximum-texture1d-width maximum-texture1d-width
   :maximum-texture2d-height maximum-texture2d-height
   :maximum-texture2d-layered-height maximum-texture2d-layered-height
   :maximum-texture2d-layered-layers maximum-texture2d-layered-layers
   :maximum-texture2d-linear-height maximum-texture2d-linear-height
   :maximum-texture2d-linear-pitch maximum-texture2d-linear-pitch
   :maximum-texture2d-linear-width maximum-texture2d-linear-width
   :maximum-texture2d-mipmapped-width maximum-texture2d-mipmapped-width
   :maximum-texture2d-mipmapped-height maximum-texture2d-mipmapped-height
   :maximum-texture2d-width maximum-texture2d-width
   :maximum-texture3d-depth maximum-texture3d-depth
   :maximum-texture3d-depth-alternate maximum-texture3d-depth-alternate
   :maximum-texture3d-height maximum-texture3d-height
   :maximum-texture3d-height-alternate maximum-texture3d-height-alternate
   :maximum-texture3d-width maximum-texture3d-width
   :maximum-texture3d-width-alternate maximum-texture3d-width-alternate
   :maximum-texturecubemap-layered-layers maximum-texturecubemap-layered-layers
   :maximum-texturecubemap-layered-width maximum-texturecubemap-layered-width
   :maximum-texturecubemap-width maximum-texturecubemap-width
   :memory-clock-rate memory-clock-rate
   :multi-gpu-board multi-gpu-board
   :multi-gpu-board-group-id multi-gpu-board-group-id
   :multiprocessor-count multiprocessor-count
   :pci-bus-id pci-bus-id
   :pci-device-id pci-device-id
   :pci-domain-id pci-domain-id
   :stream-priorities-supported stream-priorities-supported
   :surface-alignment surface-alignment
   :tcc-driver tcc-driver
   :texture-alignment texture-alignment
   :texture-pitch-alignment texture-pitch-alignment
   :total-constant-memory total-constant-memory
   :unified-addressing unified-addressing
   :warp-size warp-size})

(extend-type CUdevice
  Info
  (info
    ([d attribute]
     (if-let [attribute-fn (device-attributes attribute)]
       (maybe (attribute-fn d))
       (throw (ex-info "Unknown attribute." {:attribute attribute}))))
    ([d]
     (fmap #(maybe (% d)) device-attributes))))
