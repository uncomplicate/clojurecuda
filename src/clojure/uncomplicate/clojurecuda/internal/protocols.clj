;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.clojurecuda.internal.protocols)

;; =================== Object wrappers =================================

(defprotocol Mem
  "An object that represents memory that participates in CUDA operations.
  It can be on the device, or on the host.  Built-in implementations:
  cuda pointers, Java primitive arrays and ByteBuffers"
;; TODO remove this is not needed in JavaCPP  (ptr [this] "`Pointer` to this object.")
  (memcpy-host* [dst src size] [dst src size hstream]))

;; TODO do I need this to be in separate namespace?
(defprotocol ModuleLoad
  (module-load* [data m])
  (link-add* [data link-state type opts vals]))
