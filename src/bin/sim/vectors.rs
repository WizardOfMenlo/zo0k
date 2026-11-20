use zo0k::hash::Domain;
use zo0k::{Blake3, Field, Gf2_128, Hasher, Keccak256, Rng, blake3, keccak};

const KEY: [u8; 32] = *b"whats the Elvish word for friend";

const KECCAK_256_EMPTY: &str = "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470";

const BLAKE3_EMPTY: &str = "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262";

const KECCAK_F1600_ZERO_ONCE: [u64; 25] = [
    0xF1258F7940E1DDE7,
    0x84D5CCF933C0478A,
    0xD598261EA65AA9EE,
    0xBD1547306F80494D,
    0x8B284E056253D057,
    0xFF97A42D7F8E6FD4,
    0x90FEE5A0A44647C4,
    0x8C5BDA0CD6192E76,
    0xAD30A6F71B19059C,
    0x30935AB7D08FFC64,
    0xEB5AA93F2317D635,
    0xA9A6E6260D712103,
    0x81A57C16DBCF555F,
    0x43B831CD0347C826,
    0x01F22F1A11A5569F,
    0x05E5635A21D9AE61,
    0x64BEFEF28CC970F2,
    0x613670957BC46611,
    0xB87C5A554FD00ECB,
    0x8C3EE88A1CCF32C8,
    0x940C7922AE3A2614,
    0x1841F924A2C509E4,
    0x16F53526E70465C2,
    0x75F644E97F30A13B,
    0xEAF1FF7B5CECA249,
];

const KECCAK_F1600_ZERO_TWICE: [u64; 25] = [
    0x2D5C954DF96ECB3C,
    0x6A332CD07057B56D,
    0x093D8D1270D76B6C,
    0x8A20D9B25569D094,
    0x4F9C4F99E5E7F156,
    0xF957B9A2DA65FB38,
    0x85773DAE1275AF0D,
    0xFAF4F247C3D810F7,
    0x1F1B9EE6F79A8759,
    0xE4FECC0FEE98B425,
    0x68CE61B6B9CE68A1,
    0xDEEA66C4BA8F974F,
    0x33C43D836EAFB1F5,
    0xE00654042719DBD9,
    0x7CF8A9F009831265,
    0xFD5449A6BF174743,
    0x97DDAD33D8994B40,
    0x48EAD5FC5D0BE774,
    0xE3B8C8EE55B7B03C,
    0x91A0226E649E42E9,
    0x900E3129E7BADD7B,
    0x202A9EC5FAA3CCE8,
    0x5B3402464E1C3DB6,
    0x609F4E62A44C1059,
    0x20D06CD26A8FBF5C,
];

const DOMAINS: [Domain; 5] =
    [Domain::MerkleLeaf, Domain::MerkleNode, Domain::Transcript, Domain::Challenge, Domain::ProofOfWork];

const SPLITS: [usize; 11] = [0, 1, 7, 63, 64, 65, 135, 136, 1023, 1024, 1025];

const PIECE_LENS: [usize; 4] = [1, 7, 64, 1000];

const XOF_READ_LENS: [usize; 11] = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 89];

const KECCAK_256: [(usize, &str); 13] = [
    (0, "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470"),
    (1, "bc36789e7a1e281436464229828f817d6612f7b477d66591ff96a9e064bcc98a"),
    (63, "eed42da65350e8490c201e15dd3bdb8aeaab8618692db71db386a19b6578c59d"),
    (64, "002030bde3d4cf89919649775cd71875c4d0ab1708a380e03fefc3a28aa24831"),
    (65, "64578d7b8ae53c452c57b27375f3827854a7ead6448dc566d77a6673701f50d3"),
    (135, "cbdfd9dee5faad3818d6b06f95a219fd290b0e1706f6a82e5a595b9ce9faca62"),
    (136, "7ce759f1ab7f9ce437719970c26b0a66ff11fe3e38e17df89cf5d29c7d7f807e"),
    (137, "ac73d4fae68b8453f764007c1a20ce95994187861f0c3227a3a8e99a73a3b1db"),
    (271, "27eceb59ebc3dc8a04a5b135be641591a7278540e4556a2ba9f408194e666ec3"),
    (272, "8e2476e65823b24d96ebe239f2c1534cdf763e689e2410c3b1cb0c74e6177bfc"),
    (273, "3f02f134370e4debb95140ef49ddd3aed8c65ff1ed83a43f1b269421f179c5f9"),
    (1024, "8067fe24dad927632e32dcaf9b7958a5f301cfc4e37f419a08e059290be23370"),
    (4097, "59d5015ba6ac9c99a37af8425c89d4236df348aee58f4e9e9ddd6e75330e8611"),
];

const BLAKE3_KEYED: [(usize, &str); 12] = [
    (0, "92b2b75604ed3c761f9d6f62392c8a9227ad0ea3f09573e783f1498a4ed60d26"),
    (1, "6d7878dfff2f485635d39013278ae14f1454b8c0a3a2d34bc1ab38228a80c95b"),
    (63, "bb1eb5d4afa793c1ebdd9fb08def6c36d10096986ae0cfe148cd101170ce37ae"),
    (64, "ba8ced36f327700d213f120b1a207a3b8c04330528586f414d09f2f7d9ccb7e6"),
    (65, "c0a4edefa2d2accb9277c371ac12fcdbb52988a86edc54f0716e1591b4326e72"),
    (1023, "c951ecdf03288d0fcc96ee3413563d8a6d3589547f2c2fb36d9786470f1b9d6e"),
    (1024, "75c46f6f3d9eb4f55ecaaee480db732e6c2105546f1e675003687c31719c7ba4"),
    (1025, "357dc55de0c7e382c900fd6e320acc04146be01db6a8ce7210b7189bd664ea69"),
    (2048, "879cf1fa2ea0e79126cb1063617a05b6ad9d0b696d0d757cf053439f60a99dd1"),
    (2049, "9f29700902f7c86e514ddc4df1e3049f258b2472b6dd5267f61bf13983b78dd5"),
    (4096, "befc660aea2f1718884cd8deb9902811d332f4fc4a38cf7c7300d597a081bfc0"),
    (8193, "954a2a75420c8d6547e3ba5b98d963e6fa6491addc8c023189cc519821b4a1f5"),
];

const BLAKE3_PLAIN: [(usize, &str); 4] = [
    (0, "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262"),
    (1, "2d3adedff11b61f14c886e35afa036736dcd87a74d27b5c1510225d0f592e213"),
    (1024, "42214739f095a406f3fc83deb889744ac00df831c10daa55189b5d121c855af7"),
    (3073, "7124b49501012f81cc7f11ca069ec9226cecb8a2c850cfe644e327d22d3e1cd3"),
];

const XOF_PREFIX: &str = "357dc55de0c7e382c900fd6e320acc04146be01db6a8ce7210b7189bd664ea69362396b77fdc0d2634a552970843722066c3c15902ae5097e00ff53f1e116f1cd5352720113a837ab2452cafbde4d54085d9cf5d21ca613071551b25d52e69d6c81123872b6f19cd3bc1333edf0c52b94de23ba772cf82636cff4542540a7738d5b93052f650937d4ccc06fde34cd9b002cdb2b4ca347bf78646951682db460f75e8642508ab9add0c1f624567f7528aefd001e49d8c625b754409726de6f7e486880f6a09518e416245ef53dfd1a1260d898b25434990dbd83ff3b02da432ea4a55af778167eec37aad66d6b5f8144439fddd508da09b7897b19b8c23e9647634c85fc393d43ff845f8e49f62b0a972fe182600783d4b794faa8139dd21ab92c450bfd3ca7ceacea13f3206689b0fff2bbfcb9c3dbec717f10387c1cefdeecb";

const XOF_AT_4064: &str = "ad842d5142a3980dde0444c2c4105aa0cfd67c3ce9ca513622fb59f482e4a1ce3b884499f00d526a3494b6369a1a107b4cbcd23b0e31887a25bdf682ef05a07e3822f18452b52fe71e2c9375de0cb45ce0dbc5f915523408fa60d671896f2e59";

const RNG_STREAM: [(u8, &str); 3] = [
    (
        0,
        "a7f91ced0533c12cd59706f2dc38c2a8c39c007ae89ab6492698778c8684c48391dd91c9a9e458303b11be9646da40b527edd5dc328cf33fd205e6db357b554c0a9d6b9fbc3ff2fd60bc102049b39e54",
    ),
    (
        1,
        "5dc76cbd474e91b7e85a02bcba29892273ebf93f20bab06b2f352313a5050c0513ff1945cf4ba3136debbcccba7c460bffa2cf57d156f88ccfe1a491732b76cc18b445509e3d5aef131bff389b0dee27",
    ),
    (
        165,
        "346e01e5d27af19d0bc7972b5e2a3bd7cc1b7800e2124ae7413c12698ede9b4134d30aec6e2c689ccafeb33ab52179ed581b3a69e7b89fb40a2a35c8946eb3d32578251463130df52641c01a0009413a",
    ),
];

const RNG_BELOW: [(usize, usize); 10] = [
    (1, 0),
    (2, 1),
    (3, 1),
    (10, 0),
    (100, 99),
    (1000, 959),
    (65537, 41110),
    (1099511627776, 491845137487),
    (9223372036854775807, 4562072368790670549),
    (6, 1),
];

const HASHER_BLAKE3: [(usize, usize, &str); 15] = [
    (0, 0, "b61d3085c7ea5894c541523cae9d0b1c01034f1666d3df7eefb3a3612b9880fe"),
    (0, 3, "f41c2e5225b708ad16c11db3725e7a8f024680c63a28df6f4579395d16d15070"),
    (0, 2000, "952dd426b8ba85780dab24b57ec78a72c974f92bfe82f573449d047059d07e45"),
    (1, 0, "beec4c5e6191815845447a82c86d05adfd74c7b02fa12a82f30cddd44469edac"),
    (1, 3, "c4c55e2661e28b3b2a8937df441a2b040d4c6ad116e49c15cb53ce8a46a44dbb"),
    (1, 2000, "b36c64bfbef9372ff62ea1a6544a62a83f337d27c2d451f1668540cb1b9f1dd7"),
    (2, 0, "ed842d1c2a81dfa10aec7e9063d289e4ae64c13f3f8e040f48184f164c010e6a"),
    (2, 3, "52b0dc5c41674936ba733fc765cca1ae8cfa986fb9c853c5e98fbda942661366"),
    (2, 2000, "79d2821aa7dae024247664fed95f8d597048c62ab69fa481d531f7f74e9adb32"),
    (3, 0, "0337a545210ee3302bc7a8576c5683ee8e2e263baa11a0533d8c485d455fd541"),
    (3, 3, "4fcf1f8785f3f7cb7855219cc0d98ee60ed84994365f474312ab61df6bc9fa4b"),
    (3, 2000, "57b54db3fa4d3e473f3af50e95dbc106c225cc0abc580b6799dfced3b0709e15"),
    (4, 0, "345fa593424b1fb5c362f4e3d5ac291bbaf054a28d4d37d062cd496dc9a4f317"),
    (4, 3, "7ecceb87113a2b7f31148b2d8fb3af3b39b5a0ff708bd2657f3e00174dc78373"),
    (4, 2000, "361405f8635a6cb13af6d5a2dea1c84c39b21f1e27180d5021c92d994eab05b5"),
];

const HASHER_KECCAK256: [(usize, usize, &str); 15] = [
    (0, 0, "88fd017a5b4b1ee9731ef94afb6d2d3cda2d6cdd294b007749ffe36f907e159d"),
    (0, 3, "88f154cb8fe21b74b653988c26ab506ae502fddf82017df26eac54a290b3a24b"),
    (0, 2000, "0cd4341cf54002a83a851500dde569ae260da994820c9c94f322e6d124a1892e"),
    (1, 0, "5e1ad9b19bd0d202dc605e5019c731be9463a537f108101d9eb3522f53c0294a"),
    (1, 3, "6ec30c307b2ba1bb76ffdab3928244a5519c2086432a810c1217212854e22adf"),
    (1, 2000, "e0382d2df192383adbc8ec886793e7f4c781747417453e01e6cd353ab166fd43"),
    (2, 0, "62ca87aaaf2129f8254a2390c71c4d611e10451d11aa8c847d84441598f1b9dd"),
    (2, 3, "ba908f1511f21bcc58f13c6496bffc53f91b1d36b706244a29c727cbe4d49bc5"),
    (2, 2000, "29e7b3af9c46575b7a37c7f5b6da71fc7a02f17f538aa91a889ff49233e7b453"),
    (3, 0, "c730c1cfdcfc1996dbf8a163463136bb5143023e6ccd614c7ee6a59a17ff5784"),
    (3, 3, "6f0532d08f400270dcfa58703648fe00bda7a2b638a48e7d41db49288dea865e"),
    (3, 2000, "b12bb5147b746130e513d75d5b87a65db99a07035606cef7d3904fb217efe3f5"),
    (4, 0, "a75bc89cf2b8f2a04ab0a1c9da6e0f65e1d9231a202aca8b269b7088b0232332"),
    (4, 3, "b7b62f095ca11aa45d82e73602da20419a4f6491470619f3b803b7f37ca7dc93"),
    (4, 2000, "b546419f480b984f5043f71de5b934ef5bd43218ee04c3ac177ee8054711e3ac"),
];

pub fn check() -> Result<(), String> {
    check_published()?;
    check_keccak()?;
    check_keccak_permutation()?;
    check_blake3()?;
    check_xof()?;
    check_rng()?;
    check_hasher::<Blake3>(&HASHER_BLAKE3)?;
    check_hasher::<Keccak256>(&HASHER_KECCAK256)?;
    Ok(())
}

fn input(len_bytes: usize) -> Vec<u8> {
    let bytes: Vec<u8> = (0..len_bytes).map(|index| (index % 251) as u8).collect();
    assert!(bytes.len() == len_bytes);
    bytes
}

fn hex(bytes: &[u8]) -> String {
    let text: String = bytes.iter().map(|byte| format!("{byte:02x}")).collect();
    assert!(text.len() == 2 * bytes.len());
    text
}

fn expect(label: &str, actual: &[u8], expected: &str) -> Result<(), String> {
    assert!(!label.is_empty());
    assert!(expected.len().is_multiple_of(2));
    let actual = hex(actual);
    if actual == expected { Ok(()) } else { Err(format!("{label}: got {actual}, expected {expected}")) }
}

fn check_published() -> Result<(), String> {
    assert!(KECCAK_256[0].0 == 0 && BLAKE3_PLAIN[0].0 == 0);
    expect("keccak256 empty", &keccak::keccak256(&[]), KECCAK_256_EMPTY)?;
    expect("keccak256 empty table", &keccak::keccak256(&[]), KECCAK_256[0].1)?;
    expect("blake3 empty", &blake3::hash(&[]), BLAKE3_EMPTY)?;
    expect("blake3 empty table", &blake3::hash(&[]), BLAKE3_PLAIN[0].1)?;
    expect("blake3 hasher empty", &blake3::Hasher::new().finalize(), BLAKE3_EMPTY)?;
    expect("keccak256 hasher empty", &keccak::Keccak256::new().finalize(), KECCAK_256_EMPTY)
}

fn split_points(len_bytes: usize) -> Vec<usize> {
    let mut points: Vec<usize> = SPLITS.iter().copied().filter(|&split| split <= len_bytes).collect();
    points.push(len_bytes / 2);
    points.push(len_bytes);
    assert!(points.iter().all(|&split| split <= len_bytes));
    assert!(points.len() >= 3);
    points
}

fn check_keccak() -> Result<(), String> {
    const { assert!(keccak::RATE_BYTES == 136) };
    for (len_bytes, expected) in KECCAK_256 {
        let bytes = input(len_bytes);
        expect(&format!("keccak256 len {len_bytes}"), &keccak::keccak256(&bytes), expected)?;
        for split in split_points(len_bytes) {
            let mut hasher = keccak::Keccak256::new();
            hasher.update(&bytes[..split]);
            let snapshot = hasher.clone();
            hasher.update(&bytes[split..]);
            expect(&format!("keccak256 len {len_bytes} split {split}"), &hasher.finalize(), expected)?;
            let mut resumed = snapshot;
            resumed.update(&bytes[split..]);
            expect(&format!("keccak256 len {len_bytes} clone {split}"), &resumed.finalize(), expected)?;
        }
        for piece_len in PIECE_LENS {
            let mut hasher = keccak::Keccak256::new();
            bytes.chunks(piece_len).for_each(|piece| _ = hasher.update(piece));
            expect(&format!("keccak256 len {len_bytes} pieces {piece_len}"), &hasher.finalize(), expected)?;
        }
    }
    Ok(())
}

fn check_keccak_permutation() -> Result<(), String> {
    let mut state = [0u64; keccak::LANES];
    keccak::keccak_f1600(&mut state);
    if state != KECCAK_F1600_ZERO_ONCE {
        return Err("keccak-f[1600] of the zero state".into());
    }
    keccak::keccak_f1600(&mut state);
    if state != KECCAK_F1600_ZERO_TWICE {
        return Err("keccak-f[1600] applied twice to the zero state".into());
    }
    let mut portable = [0u64; keccak::LANES];
    keccak::keccak_f1600_portable(&mut portable);
    keccak::keccak_f1600_portable(&mut portable);
    if portable != KECCAK_F1600_ZERO_TWICE {
        return Err("portable keccak-f[1600] applied twice to the zero state".into());
    }
    let mut accelerated = KECCAK_F1600_ZERO_TWICE;
    keccak::keccak_f1600(&mut accelerated);
    keccak::keccak_f1600_portable(&mut portable);
    if accelerated != portable {
        return Err("keccak-f[1600] backends disagree".into());
    }
    Ok(())
}

fn check_blake3() -> Result<(), String> {
    for (len_bytes, expected) in BLAKE3_PLAIN {
        expect(&format!("blake3 len {len_bytes}"), &blake3::hash(&input(len_bytes)), expected)?;
    }
    for (len_bytes, expected) in BLAKE3_KEYED {
        let bytes = input(len_bytes);
        expect(&format!("blake3 keyed len {len_bytes}"), &blake3::keyed_hash(&KEY, &bytes), expected)?;
        for split in split_points(len_bytes) {
            let mut hasher = blake3::Hasher::new_keyed(&KEY);
            hasher.update(&bytes[..split]);
            let snapshot = hasher.clone();
            hasher.update(&bytes[split..]);
            expect(&format!("blake3 keyed len {len_bytes} split {split}"), &hasher.finalize(), expected)?;
            let mut resumed = snapshot;
            resumed.update(&bytes[split..]);
            expect(&format!("blake3 keyed len {len_bytes} clone {split}"), &resumed.finalize(), expected)?;
        }
        for piece_len in PIECE_LENS {
            let mut hasher = blake3::Hasher::new_keyed(&KEY);
            bytes.chunks(piece_len).for_each(|piece| _ = hasher.update(piece));
            expect(
                &format!("blake3 keyed len {len_bytes} pieces {piece_len}"),
                &hasher.finalize(),
                expected,
            )?;
        }
    }
    Ok(())
}

fn check_xof() -> Result<(), String> {
    assert!(XOF_READ_LENS.iter().sum::<usize>() * 2 == XOF_PREFIX.len());
    let mut hasher = blake3::Hasher::new_keyed(&KEY);
    hasher.update(&input(1025));
    expect("xof prefix is the hash", &hasher.finalize(), &XOF_PREFIX[..64])?;
    let mut reader = hasher.finalize_xof();
    let mut prefix = Vec::new();
    for read_len in XOF_READ_LENS {
        let mut piece = vec![0u8; read_len];
        reader.fill(&mut piece);
        prefix.extend_from_slice(&piece);
    }
    expect("xof prefix", &prefix, XOF_PREFIX)?;
    if reader.position() != prefix.len() as u64 {
        return Err(format!("xof position {} after {} bytes", reader.position(), prefix.len()));
    }
    let mut skipped = vec![0u8; 4064 - prefix.len()];
    reader.fill(&mut skipped);
    let mut middle = vec![0u8; 96];
    let (first, rest) = middle.split_at_mut(31);
    reader.fill(first);
    let (second, third) = rest.split_at_mut(33);
    reader.fill(second);
    reader.fill(third);
    expect("xof at 4064", &middle, XOF_AT_4064)?;
    let mut whole = vec![0u8; 4160];
    hasher.finalize_xof().fill(&mut whole);
    expect("xof whole prefix", &whole[..320], XOF_PREFIX)?;
    expect("xof whole at 4064", &whole[4064..], XOF_AT_4064)
}

fn check_rng() -> Result<(), String> {
    for (seed, expected) in RNG_STREAM {
        let mut rng = Rng::from_seed([seed; 32]);
        let mut stream = Vec::new();
        for _ in 0..expected.len() / 32 {
            let element: Gf2_128 = rng.field();
            let mut bytes = [0u8; 16];
            element.write_bytes(&mut bytes);
            stream.extend_from_slice(&bytes);
        }
        expect(&format!("rng seed {seed}"), &stream, expected)?;
        let mut direct = vec![0u8; stream.len()];
        blake3::Hasher::new_keyed(&[seed; 32]).finalize_xof().fill(&mut direct);
        expect(&format!("rng seed {seed} xof"), &direct, expected)?;
    }
    let mut rng = Rng::from_seed([7; 32]);
    for (bound, expected) in RNG_BELOW {
        let draw = rng.below(bound);
        if draw != expected || draw >= bound {
            return Err(format!("rng below {bound}: got {draw}, expected {expected}"));
        }
    }
    Ok(())
}

fn check_hasher<H: Hasher>(table: &[(usize, usize, &str); 15]) -> Result<(), String> {
    assert!(table.len() == DOMAINS.len() * 3);
    for &(domain_index, len_bytes, expected) in table {
        let domain = DOMAINS[domain_index];
        let bytes = input(len_bytes);
        let label = format!("{} {domain:?} len {len_bytes}", H::NAME);
        expect(&label, &H::hash(domain, &bytes), expected)?;
        for split in split_points(len_bytes) {
            let mut hasher = H::new(domain);
            hasher.update(&bytes[..split]);
            let mut other = hasher.clone();
            hasher.update(&bytes[split..]);
            expect(&format!("{label} split {split}"), &hasher.finalize(), expected)?;
            other.update(&bytes[split..]);
            other.update(&[0]);
            if hex(&other.finalize()) == expected {
                return Err(format!("{label} split {split}: an extra byte does not change the digest"));
            }
        }
    }
    Ok(())
}
