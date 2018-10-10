# Copyright 2017 UniCredit S.p.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

{.push warning[ProveInit]: off .}

import dense/dinitialize, dense/daccess, dense/dslice, dense/dequality,
  dense/dconversions, dense/diterators, dense/dcollection, dense/dtrivial_ops,
  dense/dops, dense/drow_major_ops, dense/dmixed_ops, dense/dufunc,
  dense/dstack, dense/dshared, dense/dsolvers, dense/deigenvalues, dense/ddet,
  dense/dstacking

{. pop .}
