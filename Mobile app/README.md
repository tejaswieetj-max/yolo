# Smart Glasses Companion — Flutter App

Companion app for the Context-Aware Smart Glasses navigation system (Patient / Caregiver / Doctor roles).

## What changed from the previous version

**Fixed a real compile bug:** the old `BluetoothService` class in
`services/bluetooth_service.dart` had the *same name* as a class exported
by the `flutter_blue_plus` package, which it also imported. That collision
made `discoverServices()`'s return type ambiguous and would not build.
It's renamed to `GlassesBleService` in `services/ble_service.dart`.

**The BLE connection is now real, not a mock toggle.** The Patient screen's
Connect button drives an actual `scanAndConnect()` state machine
(scanning → connecting → connected / not found) via `provider`, instead of
a button that just flips a local `bool`.

**The "glass" UI is now real glass.** Every card uses `BackdropFilter` +
`ImageFilter.blur` (see `widgets/glass_card.dart`) instead of a flat
`Colors.white.withOpacity(0.05)` box. The background (`widgets/ambient_background.dart`)
has three soft glow orbs that drift slowly via one shared `AnimationController` —
cheap enough to run behind scrolling content.

**Real animation, no extra dependency.** The old README mentioned an
`animate_do` package that was never actually added to `pubspec.yaml` or
imported anywhere — the login screen wasn't actually animated. The new
login screen has a genuine staggered fade + slide entrance built on
Flutter's own `AnimationController`/`Interval`, so there's nothing extra
to `flutter pub get`.

**Caregiver map marker is now animated**, with real concentric pulse rings
driven by an `AnimationController`, on a lightweight `CustomPaint` grid
(no Google Maps API key required — that dependency was removed from
`pubspec.yaml` since it was never used).

**Doctor's bar chart now animates in** (`swapAnimationDuration`) and has
tap tooltips.

**Patient dashboard** gained a "Recent Alerts" feed showing the kind of
hazard-classifier output the glasses hardware produces (person / vehicle /
branch, with azimuth + urgency) — ties the app back to the actual project
instead of being a generic settings screen.

## Scope, deliberately

This is the *basic* app working end-to-end, not the whole system: no
backend, no login persistence, no real telemetry over BLE (only the
sensitivity write is wired to a real characteristic). `services/location_service.dart`
is a real GPS-to-backend syncer, ready to call from the patient screen
once a backend exists — it isn't wired into the UI yet because there's
nothing on the other end for it to talk to.

## Running it

```
flutter pub get
flutter run
```

No API keys required.
