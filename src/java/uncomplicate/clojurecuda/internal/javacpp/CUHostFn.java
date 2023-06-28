//   Copyright (c) Dragan Djuric. All rights reserved.
//   The use and distribution terms for this software are covered by the
//   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
//   which can be found in the file LICENSE at the root of this distribution.
//   By using this software in any fashion, you are agreeing to be bound by
//   the terms of this license.
//   You must not remove this notice, or any other, from this software.

package uncomplicate.clojurecuda.internal.javacpp;

import clojure.lang.IFn;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.cuda.cudart.CUhostFn;
import org.bytedeco.cuda.cudart.CUstream_st;


public class CUHostFn extends CUhostFn {

    private IFn fun;

    public CUHostFn (IFn fun) {
        this.fun = fun;
    }

    public void call (Pointer userData) {
        fun.invoke(userData);
    }

}
