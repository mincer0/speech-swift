# Vendored swift-websocket

This directory is vendored from the official
[`hummingbird-project/swift-websocket`](https://github.com/hummingbird-project/swift-websocket)
repository at tag `1.6.1` (commit `126df9655565068bd97838c072c1db11f9fd42ee`).

The upstream `LICENSE.txt` and `NOTICE.txt` are retained.  The local source
change is limited to `Sources/WSCore/WebSocketHandler.swift`: its close-timeout
task is managed outside the throwing task group, explicitly cancelled, and
awaited before every close-handshake exit.  This preserves the 15-second
force-close behavior while avoiding the Swift 6.3 task deallocation crash seen
when a timeout child remained in the group.
