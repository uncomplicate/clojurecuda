;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(defproject uncomplicate/clojurecuda "0.5.1"
  :description "ClojureCUDA is a Clojure library for parallel computations with Nvidia's CUDA."
  :url "https://github.com/uncomplicate/clojurecuda"
  :scm {:name "git"
        :url "https://github.com/uncomplicate/clojurecuda"}
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.9.0"]
                 [org.jcuda/jcuda "0.9.2"]
                 [org.clojure/core.async "0.4.474"]
                 [uncomplicate/commons "0.6.0"]
                 [uncomplicate/fluokitten "0.9.0"]]

  :codox {:metadata {:doc/format :markdown}
          :src-dir-uri "http://github.com/uncomplicate/clojurecuda/blob/master/"
          :src-linenum-anchor-prefix "L"
          :output-path "docs/codox"
          :namespaces [uncomplicate.clojurecuda.core
                       uncomplicate.clojurecuda.info
                       uncomplicate.clojurecuda.toolbox
                       uncomplicate.clojurecuda.internal.protocols
                       uncomplicate.clojurecuda.internal.constants
                       uncomplicate.clojurecuda.internal.utils]}

  :profiles {:dev {:plugins [[lein-midje "3.2.1"]
                             [lein-codox "0.10.3"]]
                   :global-vars {*warn-on-reflection* true
                                 *assert* true
                                 *unchecked-math* :warn-on-boxed
                                 *print-length* 128}
                   :dependencies [[midje "1.9.2"]]
                   :jvm-opts ^:replace [#_"--add-opens=java.base/jdk.internal.ref=ALL-UNNAMED"]}}

  :javac-options ["-target" "1.8" "-source" "1.8" "-Xlint:-options"]
  :source-paths ["src/clojure" "src/cuda"]
  :test-paths ["test/clojure" "test/cuda"]
  :java-source-paths ["src/java"])
