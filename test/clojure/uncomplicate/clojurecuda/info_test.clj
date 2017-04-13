;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.clojurecuda.info-test
  (:require [midje.sweet :refer :all]
            [uncomplicate.commons.core :refer [release with-release]]
            [uncomplicate.clojurecuda
             [core :refer :all]
             [info :refer :all]])
  (:import clojure.lang.ExceptionInfo
           [java.nio ByteBuffer ByteOrder]))

(facts
 "info tests."
 (count (info (device 0))) => 82)
