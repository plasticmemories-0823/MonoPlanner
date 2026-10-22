# Third-party components

The portable distribution contains the following fixed runtime versions. Original
license files remain beside each runtime and in Python distribution metadata.
`licenses/dependencies.json` lists installed Python and frontend package versions;
`checksums.json` covers every shipped file except itself.

| Component | Version | Original license and source |
|---|---|---|
| CPython | 3.13.15 | PSF and bundled notices, `launcher/runtime/python/LICENSE.txt`; https://www.python.org/downloads/ |
| Node.js | 24.14.0 | MIT and bundled notices, `launcher/runtime/node/LICENSE`; https://nodejs.org/ |
| MinGit | 2.53.0.2 | GPLv2 and component notices, `launcher/runtime/git/LICENSE.txt` and `mingw64/share/licenses`; source https://github.com/git-for-windows/git/releases/tag/v2.53.0.windows.2 |
| WebView2 SDK | 1.0.4258.31 | Microsoft SDK notices, `launcher/runtime/LICENSE.txt` |
| WebView2 Fixed Runtime | 155.0.4283.45 | Microsoft Software License Terms, https://developer.microsoft.com/en-us/microsoft-edge/webview2/ ; runtime third-party notices remain in its folder |
| ripgrep | 15.2.0 | MIT or Unlicense; `launcher/runtime/ripgrep/COPYING`, `LICENSE-MIT`, `UNLICENSE`; https://github.com/BurntSushi/ripgrep/releases/tag/15.2.0 |
| Agent Chat UI | vendored source | MIT, `web/LICENSE` and `web/UPSTREAM.md` |

Microsoft's WebView2 SDK license and Runtime license are different. The SDK
notice does not replace the Runtime terms. The fixed Runtime is redistributed
unchanged and only serves this application. It has no automatic update guarantee;
distribute an updated application package to update this fallback Runtime.

The package includes the Git for Windows `v2.53.0.windows.2` source snapshot and
the publisher's MinGW build-source archive under `licenses/sources/`, with hashes
in the dependency manifest. The latter retains the publisher's original package
name. Both come from the linked official release and source tag.
Other backend/frontend dependency licenses remain the property of their
respective authors; this application does not relicense those dependencies.

## Backend runtime licenses

MonoPlanner-owned code is MIT. The product uses the open-source LangGraph
library; LangGraph Agent Server, its CLI and in-memory service runtime are not
included in the production dependency closure. Each shipped Python package
retains its original metadata and license files. The generated dependency
manifest is authoritative for the particular portable build.

