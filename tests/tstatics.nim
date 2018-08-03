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

import statics/initialize, statics/access, statics/iterators,
  statics/collection, statics/trivial_ops, statics/compilation,
  statics/equality, statics/ops, statics/row_major_ops, statics/mixed_ops,
  statics/solvers, statics/eigenvalues, statics/det, statics/ufunc,
  statics/slice

{. pop .}
