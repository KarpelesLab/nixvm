//! The keyboard: a US-layout XKB keymap (text v1, the format every Wayland
//! client compiles with libxkbcommon) and the modifier state that goes with
//! it.
//!
//! The compositor tracks modifiers itself from the evdev key codes the
//! embedder sends and tells clients the result in `wl_keyboard.modifiers`;
//! clients only use the keymap to turn a key code and that state into a
//! keysym. Keycodes are evdev codes + 8, as XKB's evdev rules have it.

use std::fmt::Write as _;

/// Real-modifier masks (XKB's fixed order: Shift, Lock, Control, Mod1..Mod5).
pub const SHIFT: u32 = 1 << 0;
pub const LOCK: u32 = 1 << 1;
pub const CONTROL: u32 = 1 << 2;
/// Alt.
pub const MOD1: u32 = 1 << 3;
/// Num Lock.
pub const MOD2: u32 = 1 << 4;
/// Super ("logo").
pub const MOD4: u32 = 1 << 6;

/// Each key: evdev code, XKB type (`None` lets the symbols decide), keysyms
/// by level.
const KEYS: &[(u32, Option<&str>, &[&str])] = &[
    (1, None, &["Escape"]),
    (2, None, &["1", "exclam"]),
    (3, None, &["2", "at"]),
    (4, None, &["3", "numbersign"]),
    (5, None, &["4", "dollar"]),
    (6, None, &["5", "percent"]),
    (7, None, &["6", "asciicircum"]),
    (8, None, &["7", "ampersand"]),
    (9, None, &["8", "asterisk"]),
    (10, None, &["9", "parenleft"]),
    (11, None, &["0", "parenright"]),
    (12, None, &["minus", "underscore"]),
    (13, None, &["equal", "plus"]),
    (14, None, &["BackSpace"]),
    (15, None, &["Tab", "ISO_Left_Tab"]),
    (16, Some("ALPHABETIC"), &["q", "Q"]),
    (17, Some("ALPHABETIC"), &["w", "W"]),
    (18, Some("ALPHABETIC"), &["e", "E"]),
    (19, Some("ALPHABETIC"), &["r", "R"]),
    (20, Some("ALPHABETIC"), &["t", "T"]),
    (21, Some("ALPHABETIC"), &["y", "Y"]),
    (22, Some("ALPHABETIC"), &["u", "U"]),
    (23, Some("ALPHABETIC"), &["i", "I"]),
    (24, Some("ALPHABETIC"), &["o", "O"]),
    (25, Some("ALPHABETIC"), &["p", "P"]),
    (26, None, &["bracketleft", "braceleft"]),
    (27, None, &["bracketright", "braceright"]),
    (28, None, &["Return"]),
    (29, None, &["Control_L"]),
    (30, Some("ALPHABETIC"), &["a", "A"]),
    (31, Some("ALPHABETIC"), &["s", "S"]),
    (32, Some("ALPHABETIC"), &["d", "D"]),
    (33, Some("ALPHABETIC"), &["f", "F"]),
    (34, Some("ALPHABETIC"), &["g", "G"]),
    (35, Some("ALPHABETIC"), &["h", "H"]),
    (36, Some("ALPHABETIC"), &["j", "J"]),
    (37, Some("ALPHABETIC"), &["k", "K"]),
    (38, Some("ALPHABETIC"), &["l", "L"]),
    (39, None, &["semicolon", "colon"]),
    (40, None, &["apostrophe", "quotedbl"]),
    (41, None, &["grave", "asciitilde"]),
    (42, None, &["Shift_L"]),
    (43, None, &["backslash", "bar"]),
    (44, Some("ALPHABETIC"), &["z", "Z"]),
    (45, Some("ALPHABETIC"), &["x", "X"]),
    (46, Some("ALPHABETIC"), &["c", "C"]),
    (47, Some("ALPHABETIC"), &["v", "V"]),
    (48, Some("ALPHABETIC"), &["b", "B"]),
    (49, Some("ALPHABETIC"), &["n", "N"]),
    (50, Some("ALPHABETIC"), &["m", "M"]),
    (51, None, &["comma", "less"]),
    (52, None, &["period", "greater"]),
    (53, None, &["slash", "question"]),
    (54, None, &["Shift_R"]),
    (55, None, &["KP_Multiply"]),
    (56, None, &["Alt_L", "Meta_L"]),
    (57, None, &["space"]),
    (58, None, &["Caps_Lock"]),
    (59, None, &["F1"]),
    (60, None, &["F2"]),
    (61, None, &["F3"]),
    (62, None, &["F4"]),
    (63, None, &["F5"]),
    (64, None, &["F6"]),
    (65, None, &["F7"]),
    (66, None, &["F8"]),
    (67, None, &["F9"]),
    (68, None, &["F10"]),
    (69, None, &["Num_Lock"]),
    (70, None, &["Scroll_Lock"]),
    (71, Some("KEYPAD"), &["KP_Home", "KP_7"]),
    (72, Some("KEYPAD"), &["KP_Up", "KP_8"]),
    (73, Some("KEYPAD"), &["KP_Prior", "KP_9"]),
    (74, None, &["KP_Subtract"]),
    (75, Some("KEYPAD"), &["KP_Left", "KP_4"]),
    (76, Some("KEYPAD"), &["KP_Begin", "KP_5"]),
    (77, Some("KEYPAD"), &["KP_Right", "KP_6"]),
    (78, None, &["KP_Add"]),
    (79, Some("KEYPAD"), &["KP_End", "KP_1"]),
    (80, Some("KEYPAD"), &["KP_Down", "KP_2"]),
    (81, Some("KEYPAD"), &["KP_Next", "KP_3"]),
    (82, Some("KEYPAD"), &["KP_Insert", "KP_0"]),
    (83, Some("KEYPAD"), &["KP_Delete", "KP_Decimal"]),
    (86, None, &["less", "greater"]),
    (87, None, &["F11"]),
    (88, None, &["F12"]),
    (96, None, &["KP_Enter"]),
    (97, None, &["Control_R"]),
    (98, None, &["KP_Divide"]),
    (99, None, &["Print", "Sys_Req"]),
    (100, None, &["Alt_R", "Meta_R"]),
    (102, None, &["Home"]),
    (103, None, &["Up"]),
    (104, None, &["Prior"]),
    (105, None, &["Left"]),
    (106, None, &["Right"]),
    (107, None, &["End"]),
    (108, None, &["Down"]),
    (109, None, &["Next"]),
    (110, None, &["Insert"]),
    (111, None, &["Delete"]),
    (113, None, &["XF86AudioMute"]),
    (114, None, &["XF86AudioLowerVolume"]),
    (115, None, &["XF86AudioRaiseVolume"]),
    (117, None, &["KP_Equal"]),
    (119, None, &["Pause", "Break"]),
    (125, None, &["Super_L"]),
    (126, None, &["Super_R"]),
    (127, None, &["Menu"]),
];

/// The keymap's text, NUL-terminated as `wl_keyboard.keymap` wants it.
#[must_use]
pub fn keymap() -> Vec<u8> {
    let mut s = String::from("xkb_keymap {\nxkb_keycodes \"nixvm\" {\n");
    s.push_str("  minimum = 8;\n  maximum = 255;\n");
    for (code, _, _) in KEYS {
        let _ = writeln!(s, "  <K{code}> = {};", code + 8);
    }
    s.push_str("  indicator 1 = \"Caps Lock\";\n  indicator 2 = \"Num Lock\";\n");
    s.push_str("};\n");
    s.push_str(TYPES);
    s.push_str(COMPAT);
    s.push_str("xkb_symbols \"nixvm\" {\n  name[Group1] = \"English (US)\";\n");
    for (code, ty, syms) in KEYS {
        let ty = match ty {
            Some(t) => *t,
            None if syms.len() == 1 => "ONE_LEVEL",
            None => "TWO_LEVEL",
        };
        let _ = writeln!(
            s,
            "  key <K{code}> {{ type = \"{ty}\", [ {} ] }};",
            syms.join(", ")
        );
    }
    s.push_str(
        "  modifier_map Shift { <K42>, <K54> };\n  \
         modifier_map Lock { <K58> };\n  \
         modifier_map Control { <K29>, <K97> };\n  \
         modifier_map Mod1 { <K56>, <K100> };\n  \
         modifier_map Mod2 { <K69> };\n  \
         modifier_map Mod4 { <K125>, <K126> };\n",
    );
    s.push_str("};\n};\n");
    let mut v = s.into_bytes();
    v.push(0);
    v
}

const TYPES: &str = r#"xkb_types "nixvm" {
  virtual_modifiers NumLock,Alt,Meta,Super;
  type "ONE_LEVEL" {
    modifiers = none;
    level_name[Level1] = "Any";
  };
  type "TWO_LEVEL" {
    modifiers = Shift;
    map[Shift] = Level2;
    level_name[Level1] = "Base";
    level_name[Level2] = "Shift";
  };
  type "ALPHABETIC" {
    modifiers = Shift+Lock;
    map[Shift] = Level2;
    map[Lock] = Level2;
    level_name[Level1] = "Base";
    level_name[Level2] = "Caps";
  };
  type "KEYPAD" {
    modifiers = Shift+NumLock;
    map[None] = Level1;
    map[Shift] = Level2;
    map[NumLock] = Level2;
    map[Shift+NumLock] = Level1;
    level_name[Level1] = "Base";
    level_name[Level2] = "Number";
  };
};
"#;

const COMPAT: &str = r#"xkb_compatibility "nixvm" {
  virtual_modifiers NumLock,Alt,Meta,Super;
  interpret.useModMapMods = AnyLevel;
  interpret.repeat = False;
  interpret Shift_L+AnyOf(all) { action = SetMods(modifiers=Shift,clearLocks); };
  interpret Shift_R+AnyOf(all) { action = SetMods(modifiers=Shift,clearLocks); };
  interpret Control_L+AnyOf(all) { action = SetMods(modifiers=Control,clearLocks); };
  interpret Control_R+AnyOf(all) { action = SetMods(modifiers=Control,clearLocks); };
  interpret Alt_L+AnyOf(all) { virtualModifier = Alt; action = SetMods(modifiers=modMapMods,clearLocks); };
  interpret Alt_R+AnyOf(all) { virtualModifier = Alt; action = SetMods(modifiers=modMapMods,clearLocks); };
  interpret Meta_L+AnyOf(all) { virtualModifier = Meta; action = SetMods(modifiers=modMapMods,clearLocks); };
  interpret Meta_R+AnyOf(all) { virtualModifier = Meta; action = SetMods(modifiers=modMapMods,clearLocks); };
  interpret Super_L+AnyOf(all) { virtualModifier = Super; action = SetMods(modifiers=modMapMods,clearLocks); };
  interpret Super_R+AnyOf(all) { virtualModifier = Super; action = SetMods(modifiers=modMapMods,clearLocks); };
  interpret Caps_Lock+AnyOfOrNone(all) { action = LockMods(modifiers=Lock); };
  interpret Num_Lock+AnyOf(all) { virtualModifier = NumLock; action = LockMods(modifiers=NumLock); };
  interpret Any+Exactly(Lock) { action = LockMods(modifiers=Lock); };
  interpret Any+AnyOf(all) { action = SetMods(modifiers=modMapMods,clearLocks); };
  indicator "Caps Lock" { whichModState = Locked; modifiers = Lock; };
  indicator "Num Lock" { whichModState = Locked; modifiers = NumLock; };
};
"#;

/// The modifier state, as `wl_keyboard.modifiers` reports it.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Mods {
    pub depressed: u32,
    pub latched: u32,
    pub locked: u32,
    pub group: u32,
}

impl Mods {
    /// Update for evdev key `code` going down or up, with `down` the keys
    /// held now (after this one). Returns whether anything changed.
    pub fn key(&mut self, code: u32, pressed: bool, down: &[u32]) -> bool {
        let before = *self;
        let held = |a: u32, b: u32| down.contains(&a) || down.contains(&b);
        match code {
            58 if pressed => self.locked ^= LOCK,
            69 if pressed => self.locked ^= MOD2,
            _ => {}
        }
        let mut d = 0;
        if held(42, 54) {
            d |= SHIFT;
        }
        if held(29, 97) {
            d |= CONTROL;
        }
        if held(56, 100) {
            d |= MOD1;
        }
        if held(125, 126) {
            d |= MOD4;
        }
        self.depressed = d;
        *self != before
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_keymap_names_every_key_once() {
        let text = String::from_utf8(keymap()).unwrap();
        assert!(text.ends_with("};\n\0"));
        assert!(text.contains("<K30> = 38;"), "KEY_A is keycode 38");
        assert!(text.contains("key <K30> { type = \"ALPHABETIC\", [ a, A ] };"));
        let mut codes: Vec<u32> = KEYS.iter().map(|k| k.0).collect();
        codes.dedup();
        assert_eq!(codes.len(), KEYS.len(), "keys sorted, no duplicates");
        // Braces balance, or xkbcommon rejects it.
        let open = text.matches('{').count();
        assert_eq!(open, text.matches('}').count());
    }

    #[test]
    fn modifiers_follow_the_keys() {
        let mut m = Mods::default();
        assert!(m.key(42, true, &[42]));
        assert_eq!(m.depressed, SHIFT);
        assert!(!m.key(30, true, &[42, 30]), "a letter changes nothing");
        assert!(m.key(42, false, &[30]));
        assert_eq!(m.depressed, 0);
        assert!(m.key(58, true, &[58]));
        assert_eq!(m.locked, LOCK);
        m.key(58, false, &[]);
        assert_eq!(m.locked, LOCK, "caps lock stays locked on release");
        m.key(58, true, &[58]);
        assert_eq!(m.locked, 0);
    }
}
