(function() {var type_impls = {
"arithmetic_parser":[["<details class=\"toggle implementors-toggle\" open><summary><section id=\"impl-Clone-for-LocatedSpan%3CSpan,+T%3E\" class=\"impl\"><a class=\"src rightside\" href=\"src/arithmetic_parser/spans.rs.html#24\">source</a><a href=\"#impl-Clone-for-LocatedSpan%3CSpan,+T%3E\" class=\"anchor\">§</a><h3 class=\"code-header\">impl&lt;Span: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/clone/trait.Clone.html\" title=\"trait core::clone::Clone\">Clone</a>, T: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/clone/trait.Clone.html\" title=\"trait core::clone::Clone\">Clone</a>&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/clone/trait.Clone.html\" title=\"trait core::clone::Clone\">Clone</a> for <a class=\"struct\" href=\"arithmetic_parser/struct.LocatedSpan.html\" title=\"struct arithmetic_parser::LocatedSpan\">LocatedSpan</a>&lt;Span, T&gt;</h3></section></summary><div class=\"impl-items\"><details class=\"toggle method-toggle\" open><summary><section id=\"method.clone\" class=\"method trait-impl\"><a class=\"src rightside\" href=\"src/arithmetic_parser/spans.rs.html#24\">source</a><a href=\"#method.clone\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"https://doc.rust-lang.org/nightly/core/clone/trait.Clone.html#tymethod.clone\" class=\"fn\">clone</a>(&amp;self) -&gt; <a class=\"struct\" href=\"arithmetic_parser/struct.LocatedSpan.html\" title=\"struct arithmetic_parser::LocatedSpan\">LocatedSpan</a>&lt;Span, T&gt;</h4></section></summary><div class='docblock'>Returns a copy of the value. <a href=\"https://doc.rust-lang.org/nightly/core/clone/trait.Clone.html#tymethod.clone\">Read more</a></div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.clone_from\" class=\"method trait-impl\"><span class=\"rightside\"><span class=\"since\" title=\"Stable since Rust version 1.0.0\">1.0.0</span> · <a class=\"src\" href=\"https://doc.rust-lang.org/nightly/src/core/clone.rs.html#172\">source</a></span><a href=\"#method.clone_from\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"https://doc.rust-lang.org/nightly/core/clone/trait.Clone.html#method.clone_from\" class=\"fn\">clone_from</a>(&amp;mut self, source: <a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.reference.html\">&amp;Self</a>)</h4></section></summary><div class='docblock'>Performs copy-assignment from <code>source</code>. <a href=\"https://doc.rust-lang.org/nightly/core/clone/trait.Clone.html#method.clone_from\">Read more</a></div></details></div></details>","Clone","arithmetic_parser::spans::Spanned","arithmetic_parser::spans::Location"],["<details class=\"toggle implementors-toggle\" open><summary><section id=\"impl-Debug-for-LocatedSpan%3CSpan,+T%3E\" class=\"impl\"><a class=\"src rightside\" href=\"src/arithmetic_parser/spans.rs.html#24\">source</a><a href=\"#impl-Debug-for-LocatedSpan%3CSpan,+T%3E\" class=\"anchor\">§</a><h3 class=\"code-header\">impl&lt;Span: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/fmt/trait.Debug.html\" title=\"trait core::fmt::Debug\">Debug</a>, T: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/fmt/trait.Debug.html\" title=\"trait core::fmt::Debug\">Debug</a>&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/fmt/trait.Debug.html\" title=\"trait core::fmt::Debug\">Debug</a> for <a class=\"struct\" href=\"arithmetic_parser/struct.LocatedSpan.html\" title=\"struct arithmetic_parser::LocatedSpan\">LocatedSpan</a>&lt;Span, T&gt;</h3></section></summary><div class=\"impl-items\"><details class=\"toggle method-toggle\" open><summary><section id=\"method.fmt\" class=\"method trait-impl\"><a class=\"src rightside\" href=\"src/arithmetic_parser/spans.rs.html#24\">source</a><a href=\"#method.fmt\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"https://doc.rust-lang.org/nightly/core/fmt/trait.Debug.html#tymethod.fmt\" class=\"fn\">fmt</a>(&amp;self, f: &amp;mut <a class=\"struct\" href=\"https://doc.rust-lang.org/nightly/core/fmt/struct.Formatter.html\" title=\"struct core::fmt::Formatter\">Formatter</a>&lt;'_&gt;) -&gt; <a class=\"type\" href=\"https://doc.rust-lang.org/nightly/core/fmt/type.Result.html\" title=\"type core::fmt::Result\">Result</a></h4></section></summary><div class='docblock'>Formats the value using the given formatter. <a href=\"https://doc.rust-lang.org/nightly/core/fmt/trait.Debug.html#tymethod.fmt\">Read more</a></div></details></div></details>","Debug","arithmetic_parser::spans::Spanned","arithmetic_parser::spans::Location"],["<details class=\"toggle implementors-toggle\" open><summary><section id=\"impl-From%3CLocatedSpan%3C%26str,+T%3E%3E-for-LocatedSpan%3C%26str,+T%3E\" class=\"impl\"><a class=\"src rightside\" href=\"src/arithmetic_parser/spans.rs.html#116-126\">source</a><a href=\"#impl-From%3CLocatedSpan%3C%26str,+T%3E%3E-for-LocatedSpan%3C%26str,+T%3E\" class=\"anchor\">§</a><h3 class=\"code-header\">impl&lt;'a, T&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/convert/trait.From.html\" title=\"trait core::convert::From\">From</a>&lt;<a class=\"struct\" href=\"https://docs.rs/nom_locate/~4.0.0/nom_locate/struct.LocatedSpan.html\" title=\"struct nom_locate::LocatedSpan\">LocatedSpan</a>&lt;&amp;'a <a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.str.html\">str</a>, T&gt;&gt; for <a class=\"struct\" href=\"arithmetic_parser/struct.LocatedSpan.html\" title=\"struct arithmetic_parser::LocatedSpan\">LocatedSpan</a>&lt;&amp;'a <a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.str.html\">str</a>, T&gt;</h3></section></summary><div class=\"impl-items\"><details class=\"toggle method-toggle\" open><summary><section id=\"method.from\" class=\"method trait-impl\"><a class=\"src rightside\" href=\"src/arithmetic_parser/spans.rs.html#117-125\">source</a><a href=\"#method.from\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"https://doc.rust-lang.org/nightly/core/convert/trait.From.html#tymethod.from\" class=\"fn\">from</a>(value: <a class=\"struct\" href=\"https://docs.rs/nom_locate/~4.0.0/nom_locate/struct.LocatedSpan.html\" title=\"struct nom_locate::LocatedSpan\">LocatedSpan</a>&lt;&amp;'a <a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.str.html\">str</a>, T&gt;) -&gt; Self</h4></section></summary><div class='docblock'>Converts to this type from the input type.</div></details></div></details>","From<LocatedSpan<&'a str, T>>","arithmetic_parser::spans::Spanned"],["<details class=\"toggle implementors-toggle\" open><summary><section id=\"impl-LocatedSpan%3CSpan,+T%3E\" class=\"impl\"><a class=\"src rightside\" href=\"src/arithmetic_parser/spans.rs.html#41-84\">source</a><a href=\"#impl-LocatedSpan%3CSpan,+T%3E\" class=\"anchor\">§</a><h3 class=\"code-header\">impl&lt;Span, T&gt; <a class=\"struct\" href=\"arithmetic_parser/struct.LocatedSpan.html\" title=\"struct arithmetic_parser::LocatedSpan\">LocatedSpan</a>&lt;Span, T&gt;</h3></section></summary><div class=\"impl-items\"><details class=\"toggle method-toggle\" open><summary><section id=\"method.location_offset\" class=\"method\"><a class=\"src rightside\" href=\"src/arithmetic_parser/spans.rs.html#44-46\">source</a><h4 class=\"code-header\">pub fn <a href=\"arithmetic_parser/struct.LocatedSpan.html#tymethod.location_offset\" class=\"fn\">location_offset</a>(&amp;self) -&gt; <a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.usize.html\">usize</a></h4></section></summary><div class=\"docblock\"><p>The offset represents the position of the fragment relatively to the input of the parser.\nIt starts at offset 0.</p>\n</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.location_line\" class=\"method\"><a class=\"src rightside\" href=\"src/arithmetic_parser/spans.rs.html#49-51\">source</a><h4 class=\"code-header\">pub fn <a href=\"arithmetic_parser/struct.LocatedSpan.html#tymethod.location_line\" class=\"fn\">location_line</a>(&amp;self) -&gt; <a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.u32.html\">u32</a></h4></section></summary><div class=\"docblock\"><p>The line number of the fragment relatively to the input of the parser. It starts at line 1.</p>\n</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.get_column\" class=\"method\"><a class=\"src rightside\" href=\"src/arithmetic_parser/spans.rs.html#54-56\">source</a><h4 class=\"code-header\">pub fn <a href=\"arithmetic_parser/struct.LocatedSpan.html#tymethod.get_column\" class=\"fn\">get_column</a>(&amp;self) -&gt; <a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.usize.html\">usize</a></h4></section></summary><div class=\"docblock\"><p>The column of the fragment start.</p>\n</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.fragment\" class=\"method\"><a class=\"src rightside\" href=\"src/arithmetic_parser/spans.rs.html#59-61\">source</a><h4 class=\"code-header\">pub fn <a href=\"arithmetic_parser/struct.LocatedSpan.html#tymethod.fragment\" class=\"fn\">fragment</a>(&amp;self) -&gt; <a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.reference.html\">&amp;Span</a></h4></section></summary><div class=\"docblock\"><p>The fragment that is spanned. The fragment represents a part of the input of the parser.</p>\n</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.map_extra\" class=\"method\"><a class=\"src rightside\" href=\"src/arithmetic_parser/spans.rs.html#64-72\">source</a><h4 class=\"code-header\">pub fn <a href=\"arithmetic_parser/struct.LocatedSpan.html#tymethod.map_extra\" class=\"fn\">map_extra</a>&lt;U&gt;(self, map_fn: impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/ops/function/trait.FnOnce.html\" title=\"trait core::ops::function::FnOnce\">FnOnce</a>(T) -&gt; U) -&gt; <a class=\"struct\" href=\"arithmetic_parser/struct.LocatedSpan.html\" title=\"struct arithmetic_parser::LocatedSpan\">LocatedSpan</a>&lt;Span, U&gt;</h4></section></summary><div class=\"docblock\"><p>Maps the <code>extra</code> field of this span using the provided closure.</p>\n</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.map_fragment\" class=\"method\"><a class=\"src rightside\" href=\"src/arithmetic_parser/spans.rs.html#75-83\">source</a><h4 class=\"code-header\">pub fn <a href=\"arithmetic_parser/struct.LocatedSpan.html#tymethod.map_fragment\" class=\"fn\">map_fragment</a>&lt;U&gt;(\n    self,\n    map_fn: impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/ops/function/trait.FnOnce.html\" title=\"trait core::ops::function::FnOnce\">FnOnce</a>(Span) -&gt; U,\n) -&gt; <a class=\"struct\" href=\"arithmetic_parser/struct.LocatedSpan.html\" title=\"struct arithmetic_parser::LocatedSpan\">LocatedSpan</a>&lt;U, T&gt;</h4></section></summary><div class=\"docblock\"><p>Maps the fragment field of this span using the provided closure.</p>\n</div></details></div></details>",0,"arithmetic_parser::spans::Spanned","arithmetic_parser::spans::Location"],["<details class=\"toggle implementors-toggle\" open><summary><section id=\"impl-LocatedSpan%3CSpan,+T%3E\" class=\"impl\"><a class=\"src rightside\" href=\"src/arithmetic_parser/spans.rs.html#86-113\">source</a><a href=\"#impl-LocatedSpan%3CSpan,+T%3E\" class=\"anchor\">§</a><h3 class=\"code-header\">impl&lt;Span: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Copy.html\" title=\"trait core::marker::Copy\">Copy</a>, T&gt; <a class=\"struct\" href=\"arithmetic_parser/struct.LocatedSpan.html\" title=\"struct arithmetic_parser::LocatedSpan\">LocatedSpan</a>&lt;Span, T&gt;</h3></section></summary><div class=\"impl-items\"><details class=\"toggle method-toggle\" open><summary><section id=\"method.as_ref\" class=\"method\"><a class=\"src rightside\" href=\"src/arithmetic_parser/spans.rs.html#88-96\">source</a><h4 class=\"code-header\">pub fn <a href=\"arithmetic_parser/struct.LocatedSpan.html#tymethod.as_ref\" class=\"fn\">as_ref</a>(&amp;self) -&gt; <a class=\"struct\" href=\"arithmetic_parser/struct.LocatedSpan.html\" title=\"struct arithmetic_parser::LocatedSpan\">LocatedSpan</a>&lt;Span, <a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.reference.html\">&amp;T</a>&gt;</h4></section></summary><div class=\"docblock\"><p>Returns a copy of this span with borrowed <code>extra</code> field.</p>\n</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.copy_with_extra\" class=\"method\"><a class=\"src rightside\" href=\"src/arithmetic_parser/spans.rs.html#99-107\">source</a><h4 class=\"code-header\">pub fn <a href=\"arithmetic_parser/struct.LocatedSpan.html#tymethod.copy_with_extra\" class=\"fn\">copy_with_extra</a>&lt;U&gt;(&amp;self, value: U) -&gt; <a class=\"struct\" href=\"arithmetic_parser/struct.LocatedSpan.html\" title=\"struct arithmetic_parser::LocatedSpan\">LocatedSpan</a>&lt;Span, U&gt;</h4></section></summary><div class=\"docblock\"><p>Copies this span with the provided <code>extra</code> field.</p>\n</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.with_no_extra\" class=\"method\"><a class=\"src rightside\" href=\"src/arithmetic_parser/spans.rs.html#110-112\">source</a><h4 class=\"code-header\">pub fn <a href=\"arithmetic_parser/struct.LocatedSpan.html#tymethod.with_no_extra\" class=\"fn\">with_no_extra</a>(&amp;self) -&gt; <a class=\"struct\" href=\"arithmetic_parser/struct.LocatedSpan.html\" title=\"struct arithmetic_parser::LocatedSpan\">LocatedSpan</a>&lt;Span&gt;</h4></section></summary><div class=\"docblock\"><p>Removes <code>extra</code> field from this span.</p>\n</div></details></div></details>",0,"arithmetic_parser::spans::Spanned","arithmetic_parser::spans::Location"],["<details class=\"toggle implementors-toggle\" open><summary><section id=\"impl-PartialEq-for-LocatedSpan%3CSpan,+T%3E\" class=\"impl\"><a class=\"src rightside\" href=\"src/arithmetic_parser/spans.rs.html#35-39\">source</a><a href=\"#impl-PartialEq-for-LocatedSpan%3CSpan,+T%3E\" class=\"anchor\">§</a><h3 class=\"code-header\">impl&lt;Span: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/cmp/trait.PartialEq.html\" title=\"trait core::cmp::PartialEq\">PartialEq</a>, T&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/cmp/trait.PartialEq.html\" title=\"trait core::cmp::PartialEq\">PartialEq</a> for <a class=\"struct\" href=\"arithmetic_parser/struct.LocatedSpan.html\" title=\"struct arithmetic_parser::LocatedSpan\">LocatedSpan</a>&lt;Span, T&gt;</h3></section></summary><div class=\"impl-items\"><details class=\"toggle method-toggle\" open><summary><section id=\"method.eq\" class=\"method trait-impl\"><a class=\"src rightside\" href=\"src/arithmetic_parser/spans.rs.html#36-38\">source</a><a href=\"#method.eq\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"https://doc.rust-lang.org/nightly/core/cmp/trait.PartialEq.html#tymethod.eq\" class=\"fn\">eq</a>(&amp;self, other: <a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.reference.html\">&amp;Self</a>) -&gt; <a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.bool.html\">bool</a></h4></section></summary><div class='docblock'>This method tests for <code>self</code> and <code>other</code> values to be equal, and is used\nby <code>==</code>.</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.ne\" class=\"method trait-impl\"><span class=\"rightside\"><span class=\"since\" title=\"Stable since Rust version 1.0.0\">1.0.0</span> · <a class=\"src\" href=\"https://doc.rust-lang.org/nightly/src/core/cmp.rs.html#262\">source</a></span><a href=\"#method.ne\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"https://doc.rust-lang.org/nightly/core/cmp/trait.PartialEq.html#method.ne\" class=\"fn\">ne</a>(&amp;self, other: <a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.reference.html\">&amp;Rhs</a>) -&gt; <a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.bool.html\">bool</a></h4></section></summary><div class='docblock'>This method tests for <code>!=</code>. The default implementation is almost always\nsufficient, and should not be overridden without very good reason.</div></details></div></details>","PartialEq","arithmetic_parser::spans::Spanned","arithmetic_parser::spans::Location"],["<section id=\"impl-Copy-for-LocatedSpan%3CSpan,+T%3E\" class=\"impl\"><a class=\"src rightside\" href=\"src/arithmetic_parser/spans.rs.html#24\">source</a><a href=\"#impl-Copy-for-LocatedSpan%3CSpan,+T%3E\" class=\"anchor\">§</a><h3 class=\"code-header\">impl&lt;Span: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Copy.html\" title=\"trait core::marker::Copy\">Copy</a>, T: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Copy.html\" title=\"trait core::marker::Copy\">Copy</a>&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Copy.html\" title=\"trait core::marker::Copy\">Copy</a> for <a class=\"struct\" href=\"arithmetic_parser/struct.LocatedSpan.html\" title=\"struct arithmetic_parser::LocatedSpan\">LocatedSpan</a>&lt;Span, T&gt;</h3></section>","Copy","arithmetic_parser::spans::Spanned","arithmetic_parser::spans::Location"]]
};if (window.register_type_impls) {window.register_type_impls(type_impls);} else {window.pending_type_impls = type_impls;}})()