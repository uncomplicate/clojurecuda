;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(defproject uncomplicate/clojurecuda "0.22.1"
  :description "ClojureCUDA is a Clojure library for parallel computations with Nvidia's CUDA."
  :url "https://github.com/uncomplicate/clojurecuda"
  :scm {:name "git"
        :url "https://github.com/uncomplicate/clojurecuda"}
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.12.1"]
                 [org.clojure/core.async "1.8.741"]
                 [uncomplicate/commons "0.17.1"]
                 [uncomplicate/fluokitten "0.10.0"]
                 [org.uncomplicate/clojure-cpp "0.5.1"]
                 [org.bytedeco/cuda-platform "12.9-9.10-1.5.12"]]

  :profiles {:dev [:dev/all ~(leiningen.core.utils/get-os)]
             :dev/all {:plugins [[lein-midje "3.2.1"]
                                 [lein-codox "0.10.8"]
                                 [com.github.clj-kondo/lein-clj-kondo "0.2.5"]]
                       :global-vars {*warn-on-reflection* true
                                     *assert* true
                                     *unchecked-math* :warn-on-boxed
                                     *print-length* 128}
                       :dependencies [[midje "1.10.10"]
                                      [codox-theme-rdash "0.1.2"]]
                       :codox {:metadata {:doc/format :markdown}
                               :source-uri "http://github.com/uncomplicate/clojurecuda/blob/master/{filepath}#L{line}"
                               :output-path "docs/codox"
                               :themes [:rdash]
                               :namespaces [uncomplicate.clojurecuda.core
                                            uncomplicate.clojurecuda.info
                                            uncomplicate.clojurecuda.toolbox
                                            uncomplicate.clojurecuda.internal.constants]}
                       :jvm-opts ^:replace ["-Dclojure.compiler.direct-linking=true"
                                            "--enable-native-access=ALL-UNNAMED"]}
             :linux {:dependencies [[org.bytedeco/cuda "12.9-9.10-1.5.12-20250612.143830-1" :classifier "linux-x86_64-redist"]]}
             :windows {:dependencies [[org.bytedeco/cuda "12.9-9.10-1.5.12-20250612.145546-3" :classifier "windows-x86_64-redist"]]}}

  :repositories [["snapshots" "https://oss.sonatype.org/content/repositories/snapshots"]]

  :javac-options ["-target" "1.8" "-source" "1.8" "-Xlint:-options"]

  :source-paths ["src/clojure" "src/cuda"]
  :test-paths ["test/clojure" "test/cuda"]
  :java-source-paths ["src/java"])
