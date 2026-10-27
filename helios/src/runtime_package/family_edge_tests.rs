use super::*;
use crate::ReplayWorker;
use std::time::Duration;

/// Local diagnostic (MiMo PDL ordering bug): replay a single-GPU family package exported by
/// pdl-tools/family_export.py and compare its "output" buffer with the eager reference after
/// every replay. FLARE_GRAPH_PROGRAMMATIC_EDGES selects exact (unset) or serialized (off) edges.
#[test]
#[ignore = "requires FLARE_FAMILY_EDGE_TEST_PACKAGE exported by family_export.py on the assigned GPU"]
fn family_package_replays_match_eager_reference() {
    let root = std::path::PathBuf::from(std::env::var_os("FLARE_FAMILY_EDGE_TEST_PACKAGE").unwrap());
    let expected = std::fs::read(root.join("expected.bin")).unwrap();
    let replays: u32 = std::env::var("FLARE_FAMILY_EDGE_REPLAYS").map_or(200, |value| value.parse().unwrap());
    let owner = unsafe { NativeResources::load(test_support::authorize(root.join("package")), 0).unwrap() };
    let buffer = owner
        .execution
        .as_ref()
        .and_then(|execution| execution.buffers.iter().find(|buffer| buffer.name == "output"))
        .expect("package exports an output buffer");
    let reference = buffer.reference.as_ref().unwrap();
    let offset = protobuf::size(reference.offset).unwrap();
    assert_eq!(protobuf::size(buffer.nbytes).unwrap(), expected.len());
    let graph = owner.graph_names().iter().position(|name| name == "decode").unwrap();
    let worker = unsafe { ReplayWorker::new(owner.graph_bundle(), 1).unwrap() };
    let mut mismatches = 0;
    for replay in 0..replays {
        worker.submit(graph, 1).unwrap();
        let completed = worker.poll(Duration::from_secs(30)).unwrap().unwrap();
        assert!(completed.error.is_none(), "{:?}", completed.error);
        let bytes = owner.download(&reference.allocation).unwrap();
        if bytes[offset..offset + expected.len()] != expected[..] {
            mismatches += 1;
            if mismatches <= 3 {
                eprintln!("replay {replay}: output differs from the eager reference");
                std::fs::write(root.join(format!("mismatch-{replay}.bin")), &bytes[offset..offset + expected.len()])
                    .unwrap();
            }
        }
    }
    assert!(worker.close().drained);
    eprintln!("family replays: {replays}, mismatches: {mismatches}");
    assert_eq!(mismatches, 0, "{mismatches} of {replays} replays differ from the eager reference");
}
