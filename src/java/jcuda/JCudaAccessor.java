//   Copyright (c) Dragan Djuric. All rights reserved.
//   The use and distribution terms for this software are covered by the
//   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
//   which can be found in the file LICENSE at the root of this distribution.
//   By using this software in any fashion, you are agreeing to be bound by
//   the terms of this license.
//   You must not remove this notice, or any other, from this software.

package jcuda;

public class JCudaAccessor {

    private static class ZeroPointer extends NativePointerObject {
        public ZeroPointer() {
            super();
        }
    }

    public static int zeroHash = new ZeroPointer().hashCode();

    public static boolean isNullPointer (NativePointerObject npo) {
        return zeroHash == npo.hashCode();
    }

    public static long getNativePointer (NativePointerObject npo) {
        return npo.getNativePointer();
    };
}
