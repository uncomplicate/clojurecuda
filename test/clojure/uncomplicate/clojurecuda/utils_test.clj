;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.clojurecuda.utils-test
  (:require [midje.sweet :refer :all]
            [uncomplicate.clojurecuda.internal.utils :refer :all]))

(facts
 "error tests"

 (ex-data (error 0))
 => {:code 0, :details nil, :name :success :type :cuda}

 (ex-data (error -43))
 => {:code -43, :details nil, :name -43, :type :cuda}

 (ex-data (error 0 "Additional details"))
 => {:code 0, :details "Additional details", :name :success, :type :cuda})

(facts
 "with-check tests"
 (let [f (fn [x] (if x 0 -1))]
   (with-check (f 1) :success) => :success
   (with-check (f false) :success) => (throws clojure.lang.ExceptionInfo)))

(facts
 "maybe tests"
 (ex-data (maybe (throw (ex-info "Test Exception" {:data :test}))))
 => (throws clojure.lang.ExceptionInfo)

 (:type (ex-data (error -1 nil))) => :cuda)
