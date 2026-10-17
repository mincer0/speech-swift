import Foundation
import XCTest
#if canImport(Darwin)
import Darwin
#endif
@testable import MiniCPMDemoBackend

final class MiniCPMDirectorySizeTests: XCTestCase {
    private func withFixture(_ body: (URL) throws -> Void) throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("MiniCPM 目录大小 \(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try body(root)
    }

    private func write(_ bytes: Int, to file: URL) throws {
        try FileManager.default.createDirectory(
            at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data(repeating: 0x61, count: bytes).write(to: file)
    }

    func testCountsVisibleRegularFilesRecursivelyAndIgnoresDotEntriesAndEmptyDirectories() throws {
        try withFixture { root in
            try write(11, to: root.appendingPathComponent("可见 文件.bin"))
            try write(23, to: root.appendingPathComponent("嵌套 目录/第二个 文件.dat"))
            try write(0, to: root.appendingPathComponent("empty file"))
            try FileManager.default.createDirectory(
                at: root.appendingPathComponent("空 目录/更深空目录", isDirectory: true),
                withIntermediateDirectories: true)
            try write(101, to: root.appendingPathComponent(".dot-file"))
            try write(211, to: root.appendingPathComponent(".dot-directory/visible child.bin"))
            try write(307, to: root.appendingPathComponent("嵌套 目录/.nested-hidden"))

            XCTAssertEqual(MiniCPMDemoPersistence.directorySize(root), 34)
            XCTAssertEqual(MiniCPMDemoPersistence.directorySize(
                root.appendingPathComponent("空 目录", isDirectory: true)), 0)
        }
    }

    #if os(macOS)
    func testFinderHiddenFlagsExcludeFilesAndWholeDirectorySubtrees() throws {
        try withFixture { root in
            try write(5, to: root.appendingPathComponent("visible.bin"))
            try write(7, to: root.appendingPathComponent("visible directory/nested.bin"))
            let hiddenFile = root.appendingPathComponent("Finder hidden file.bin")
            let hiddenDirectory = root.appendingPathComponent("Finder hidden directory", isDirectory: true)
            try write(109, to: hiddenFile)
            try write(223, to: hiddenDirectory.appendingPathComponent("visible-name child.bin"))
            for hidden in [hiddenFile, hiddenDirectory] {
                let result = hidden.withUnsafeFileSystemRepresentation { path in
                    chflags(path!, UInt32(UF_HIDDEN))
                }
                XCTAssertEqual(result, 0, "chflags failed for \(hidden.path), errno=\(errno)")
                XCTAssertEqual(try hidden.resourceValues(forKeys: [.isHiddenKey]).isHidden, true,
                               "fixture must be Finder-hidden without relying on a dot prefix")
            }

            XCTAssertEqual(MiniCPMDemoPersistence.directorySize(root), 12)
        }
    }
    #endif

    func testSymlinksToFilesDirectoriesAndMissingTargetsAreNotCountedOrTraversed() throws {
        try withFixture { fixture in
            let root = fixture.appendingPathComponent("计量 目录", isDirectory: true)
            let file = root.appendingPathComponent("original.bin")
            let directory = root.appendingPathComponent("nested", isDirectory: true)
            try write(13, to: file)
            try write(17, to: directory.appendingPathComponent("child.bin"))
            let external = fixture.appendingPathComponent("计量范围外", isDirectory: true)
            try write(401, to: external.appendingPathComponent("external.bin"))
            let links: [(String, URL)] = [
                ("file link", file),
                ("directory link", directory),
                ("external file link", external.appendingPathComponent("external.bin")),
                ("external directory link", external),
                ("missing target link", fixture.appendingPathComponent("does not exist")),
                ("ancestor cycle link", root),
            ]
            for (name, target) in links {
                try FileManager.default.createSymbolicLink(
                    at: root.appendingPathComponent(name), withDestinationURL: target)
            }

            XCTAssertEqual(MiniCPMDemoPersistence.directorySize(root), 30,
                           "links neither duplicate live files nor include external targets")
            let rootLink = fixture.appendingPathComponent("root link")
            try FileManager.default.createSymbolicLink(at: rootLink, withDestinationURL: root)
            XCTAssertEqual(MiniCPMDemoPersistence.directorySize(rootLink), 0,
                           "a symlink passed as the root must not be followed either")
        }
    }

    func testEachCallReflectsFileLengthChangesAdditionsAndDeletions() throws {
        try withFixture { root in
            let first = root.appendingPathComponent("first.bin")
            let second = root.appendingPathComponent("new directory/second.bin")
            try write(19, to: first)
            XCTAssertEqual(MiniCPMDemoPersistence.directorySize(root), 19)
            try write(43, to: first)
            XCTAssertEqual(MiniCPMDemoPersistence.directorySize(root), 43,
                           "growing an existing path must update the next quota calculation")
            try write(3, to: first)
            XCTAssertEqual(MiniCPMDemoPersistence.directorySize(root), 3,
                           "truncating an existing path must not retain its former size")
            try write(29, to: second)
            XCTAssertEqual(MiniCPMDemoPersistence.directorySize(root), 32)
            try FileManager.default.removeItem(at: first)
            XCTAssertEqual(MiniCPMDemoPersistence.directorySize(root), 29)
            try FileManager.default.removeItem(at: second.deletingLastPathComponent())
            XCTAssertEqual(MiniCPMDemoPersistence.directorySize(root), 0)
        }
    }

    func testMissingAndOrdinaryFileRootsHaveNoDirectoryContents() throws {
        try withFixture { root in
            let missing = root.appendingPathComponent("不存在 目录", isDirectory: true)
            XCTAssertEqual(MiniCPMDemoPersistence.directorySize(missing), 0)
            let file = root.appendingPathComponent("普通 文件.bin")
            try write(73, to: file)
            XCTAssertEqual(MiniCPMDemoPersistence.directorySize(file), 0,
                           "Foundation directory enumeration does not count a file root itself")
        }
    }
}
