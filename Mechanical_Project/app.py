"""Pelton wheel basic sizing and hydraulic power estimator.

Run locally with:
    streamlit run app.py
"""

from __future__ import annotations

from dataclasses import dataclass
from textwrap import dedent

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

G = 9.81  # m/s^2
RHO = 1000.0  # kg/m^3, fresh water

GROUP_NUMBER = 25
MEMBERS = (
    ("CHAUHAN KARTIK ASHOKBHAI", "25012250610075"),
    ("KANTARIYA SHLOK RAKESHBHAI", "25012251210005"),
    ("PATEL MAHIL CHIRAGBHAI", "25012250610052"),
    ("SOLANKI TAKSHA RASHMIKANTBHAI", "25012250610066"),
)


@dataclass(frozen=True)
class Inputs:
    gross_head_m: float
    penstock_loss_m: float
    flow_m3s: float
    runner_speed_rpm: float
    velocity_coefficient: float
    speed_ratio: float
    jet_count: int


@dataclass(frozen=True)
class Results:
    net_head_m: float
    jet_velocity_ms: float
    bucket_speed_ms: float
    hydraulic_power_kw: float
    pitch_diameter_m: float
    jet_diameter_m: float
    jet_ratio: float
    specific_speed: float


def evaluate(data: Inputs) -> Results:
    """Size a single operating point from head, loss, flow, and runner speed."""
    net_head = data.gross_head_m - data.penstock_loss_m
    jet_velocity = float(data.velocity_coefficient * np.sqrt(2.0 * G * net_head))
    bucket_speed = float(data.speed_ratio * jet_velocity)
    hydraulic_power = float(RHO * G * data.flow_m3s * net_head / 1000.0)
    pitch_diameter = float(60.0 * bucket_speed / (np.pi * data.runner_speed_rpm))
    jet_diameter = float(
        np.sqrt((4.0 * data.flow_m3s) / (data.jet_count * np.pi * jet_velocity))
    )
    specific_speed = float(
        data.runner_speed_rpm * np.sqrt(hydraulic_power) / (net_head ** 1.25)
    )
    return Results(
        net_head_m=net_head,
        jet_velocity_ms=jet_velocity,
        bucket_speed_ms=bucket_speed,
        hydraulic_power_kw=hydraulic_power,
        pitch_diameter_m=pitch_diameter,
        jet_diameter_m=jet_diameter,
        jet_ratio=pitch_diameter / jet_diameter,
        specific_speed=specific_speed,
    )


def validate(data: Inputs) -> tuple[list[str], list[str]]:
    """Reject impossible inputs, and flag design values outside usual Pelton practice."""
    errors: list[str] = []
    warnings: list[str] = []

    if data.gross_head_m <= 0:
        errors.append("Gross head must be greater than zero.")
    if data.penstock_loss_m < 0:
        errors.append("Penstock loss cannot be negative. Enter the head drop as a positive value.")
    if data.flow_m3s <= 0:
        errors.append("Water flow rate must be greater than zero.")
    if data.runner_speed_rpm <= 0:
        errors.append("Runner speed must be greater than zero.")

    if errors:
        return errors, warnings

    if data.gross_head_m - data.penstock_loss_m <= 0:
        errors.append(
            "Penstock loss must be smaller than the gross head so the net head stays positive."
        )
        return errors, warnings

    net_head = data.gross_head_m - data.penstock_loss_m
    if net_head < 15:
        warnings.append(
            "Net head is under 15 m. A Pelton wheel is normally chosen for a high head, "
            "so check the gross head and the penstock loss."
        )
    if not 0.44 <= data.speed_ratio <= 0.48:
        warnings.append(
            "Speed ratio is outside the usual Pelton band of about 0.45 to 0.47. "
            "Bucket speed may sit away from the best-efficiency point."
        )
    if data.velocity_coefficient < 0.95:
        warnings.append(
            "Velocity coefficient is below 0.95. Pelton nozzles are usually about 0.97 to 0.99."
        )

    results = evaluate(data)
    if results.jet_ratio < 9:
        warnings.append(
            f"Jet ratio D/d is {results.jet_ratio:.1f}, below the usual range of about 9 to 20. "
            "Lower the runner speed, add jets, or reduce the flow rate."
        )
    elif results.jet_ratio > 20:
        warnings.append(
            f"Jet ratio D/d is {results.jet_ratio:.1f}, above the usual range of about 9 to 20. "
            "Raise the runner speed or use fewer jets."
        )

    per_jet_limit = 30.0 * np.sqrt(data.jet_count)
    if results.specific_speed < 8 or results.specific_speed > per_jet_limit:
        warnings.append(
            f"Metric specific speed is {results.specific_speed:.1f}. "
            f"Single-jet Pelton wheels are usually near 8 to 30 "
            f"(about {per_jet_limit:.0f} for {data.jet_count} jet"
            f"{'s' if data.jet_count != 1 else ''}, with P in kW). "
            "Revisit head, power, and runner speed before treating this as a final size."
        )
    return errors, warnings


def build_figure(data: Inputs, results: Results) -> plt.Figure:
    """Power-flow lines and speed-head curves, with the operating point marked."""
    q_max = max(data.flow_m3s * 2.0, 0.2)
    flow = np.linspace(0.0, q_max, 240)
    h_max = max(results.net_head_m * 1.6, 40.0)
    head = np.linspace(0.0, h_max, 240)
    jet_speed = data.velocity_coefficient * np.sqrt(2.0 * G * head)
    bucket_speed = data.speed_ratio * jet_speed

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.5))
    power_ax, speed_ax = axes

    curve_styles = (
        (0.65, "--", "#0072B2", 1.4),
        (1.0, "-", "#D55E00", 2.2),
        (1.35, "--", "#009E73", 1.4),
    )
    for factor, style, color, width in curve_styles:
        head_case = results.net_head_m * factor
        power = RHO * G * flow * head_case / 1000.0
        label = f"Net head {head_case:.1f} m"
        if factor == 1.0:
            label = f"Current net head {head_case:.1f} m"
        power_ax.plot(flow, power, linestyle=style, color=color, linewidth=width, label=label)

    power_ax.scatter(
        [data.flow_m3s],
        [results.hydraulic_power_kw],
        s=55,
        color="#D55E00",
        zorder=5,
        label="Operating point",
    )
    power_ax.set_title("Hydraulic power vs flow rate")
    power_ax.set_xlabel(r"Flow rate (m$^3$/s)")
    power_ax.set_ylabel("Hydraulic power (kW)")
    power_ax.set_xlim(0.0, q_max)
    power_ax.set_ylim(bottom=0.0)
    power_ax.legend(loc="upper left", framealpha=0.95)
    _style_axis(power_ax)

    speed_ax.plot(head, jet_speed, color="#0072B2", linewidth=2.2, label="Jet velocity")
    speed_ax.plot(head, bucket_speed, color="#E69F00", linewidth=2.2, label="Bucket speed")
    speed_ax.scatter(
        [results.net_head_m],
        [results.jet_velocity_ms],
        s=55,
        color="#0072B2",
        zorder=5,
        label="Operating point",
    )
    speed_ax.scatter(
        [results.net_head_m],
        [results.bucket_speed_ms],
        s=55,
        color="#E69F00",
        zorder=5,
    )
    speed_ax.set_title("Jet and bucket speed vs net head")
    speed_ax.set_xlabel("Net head (m)")
    speed_ax.set_ylabel("Speed (m/s)")
    speed_ax.set_xlim(0.0, h_max)
    speed_ax.set_ylim(bottom=0.0)
    speed_ax.legend(loc="upper left", framealpha=0.95)
    _style_axis(speed_ax)

    fig.tight_layout()
    return fig


def _style_axis(axis: plt.Axes) -> None:
    axis.minorticks_on()
    axis.grid(True, which="major", linestyle="-", linewidth=0.6, alpha=0.4)
    axis.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.28)
    axis.set_axisbelow(True)


def _spin_seconds(runner_speed_rpm: float) -> float:
    """Map runner speed to a visible wheel animation. This is a visual cue only."""
    if runner_speed_rpm <= 0:
        return 2.4
    return float(np.clip(180.0 / runner_speed_rpm, 0.7, 3.5))


def _wheel_markup() -> str:
    buckets = "".join("<span>&#8203;</span>" for _ in range(10))
    return (
        '<div class="turbine" aria-hidden="true">'
        '<div class="jet">&#8203;</div>'
        '<div class="nozzle">&#8203;</div>'
        f'<div class="wheel">{buckets}</div>'
        "</div>"
    )


def _visual_drive(data: Inputs) -> dict[str, float | bool | str]:
    """Jet force and wheel speed for the diagram only. Reported results stay unchanged."""
    net_head = data.gross_head_m - data.penstock_loss_m
    has_water = net_head > 0 and data.flow_m3s > 0
    if has_water:
        jet_velocity = float(data.velocity_coefficient * np.sqrt(2.0 * G * net_head))
        force_n = float(RHO * data.flow_m3s * jet_velocity)
    else:
        jet_velocity = 0.0
        force_n = 0.0
    reference_force = float(RHO * 0.35 * (0.98 * np.sqrt(2.0 * G * 250.0)))
    spinning = has_water and data.runner_speed_rpm > 0
    if spinning:
        ratio = force_n / reference_force
        visual_rpm = data.runner_speed_rpm * (0.70 + 0.30 * float(np.clip(ratio, 0.25, 2.5)))
        period = float(np.clip(480.0 / visual_rpm, 0.18, 6.0))
    else:
        period = 0.0
    jet_seconds = float(np.clip(24.0 / max(jet_velocity, 1.0), 0.16, 1.8)) if has_water else 0.0
    drop_px = float(np.clip(5.0 + max(data.flow_m3s, 0.0) * 22.0, 4.0, 18.0))
    if force_n >= 1000:
        force_text = f"{force_n / 1000.0:.1f} kN"
    else:
        force_text = f"{force_n:.0f} N"
    if abs(data.runner_speed_rpm - round(data.runner_speed_rpm)) < 1e-6:
        rpm_text = f"{data.runner_speed_rpm:.0f}"
    else:
        rpm_text = f"{data.runner_speed_rpm:.1f}"
    return {
        "has_water": has_water,
        "spinning": spinning,
        "period": period,
        "jet_seconds": jet_seconds,
        "drop_px": drop_px,
        "force_text": force_text,
        "rpm_text": rpm_text,
        "beam_opacity": float(np.clip(0.25 + force_n / reference_force * 0.55, 0.0, 0.95)) if has_water else 0.0,
    }


_PELTON_SCRIPT = r"""
<script>
(function () {
  var root = document.getElementById("pv-viewport");
  if (!root) return;
  var token = (window.__peltonToken || 0) + 1;
  window.__peltonToken = token;
  if (window.__peltonCleanup) window.__peltonCleanup();

  var period = parseFloat(root.getAttribute("data-period") || "0");
  var jet = parseFloat(root.getAttribute("data-jet") || "0.4");
  var spinning = root.getAttribute("data-spinning") === "1";
  var water = root.getAttribute("data-water") === "1";
  var dropCount = parseInt(root.getAttribute("data-drops") || "8", 10);
  var dropPx = parseFloat(root.getAttribute("data-drop") || "8");
  if (window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches) spinning = false;

  function showNote(message) {
    if (window.__peltonToken !== token) return;
    var note = document.createElement("p");
    note.className = "pv-note";
    note.textContent = message;
    root.appendChild(note);
  }

  function boot() {
    if (window.__peltonToken !== token || !window.THREE) return;
    var THREE = window.THREE;
    var renderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    } catch (error) {
      showNote("This browser could not start the 3D view.");
      return;
    }
    renderer.setClearColor(0xffffff, 1);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    var canvas = renderer.domElement;
    root.insertBefore(canvas, root.firstChild);
    window.__peltonRenderer = renderer;

    var scene = new THREE.Scene();
    var camera = new THREE.PerspectiveCamera(40, 1, 0.1, 40);
    var look = new THREE.Vector3(-0.42, -0.06, -0.15);
    scene.add(new THREE.HemisphereLight(0xffffff, 0xd5dee8, 0.9));
    scene.add(new THREE.AmbientLight(0xffffff, 0.25));
    var sun = new THREE.DirectionalLight(0xffffff, 1.2);
    sun.position.set(5, 8, 6);
    scene.add(sun);
    var fill = new THREE.DirectionalLight(0xd7e7ff, 0.45);
    fill.position.set(-6, 3, -4);
    scene.add(fill);

    var bucketMat = new THREE.MeshStandardMaterial({ color: 0x1f5fbf, metalness: 0.48, roughness: 0.32, side: THREE.DoubleSide });
    var bowlMat = new THREE.MeshStandardMaterial({ color: 0xd7e8fb, metalness: 0.12, roughness: 0.38, side: THREE.DoubleSide });
    var hubMat = new THREE.MeshStandardMaterial({ color: 0xd5dde6, metalness: 0.72, roughness: 0.28 });
    var darkMat = new THREE.MeshStandardMaterial({ color: 0x24282e, metalness: 0.4, roughness: 0.42 });
    var pipeMat = new THREE.MeshStandardMaterial({ color: 0x5d6c7b, metalness: 0.55, roughness: 0.38 });
    var nozzleMat = new THREE.MeshStandardMaterial({ color: 0xf4f7fb, metalness: 0.28, roughness: 0.36 });
    var waterMat = new THREE.MeshStandardMaterial({ color: 0x3aa6ea, metalness: 0.05, roughness: 0.16, transparent: true, opacity: 0.92 });
    var tailMat = new THREE.MeshStandardMaterial({ color: 0x8ecff5, metalness: 0.02, roughness: 0.2, transparent: true, opacity: 0.88 });

    var wheel = new THREE.Group();
    var shellGeo = new THREE.SphereGeometry(0.22, 16, 12, 0.2, Math.PI * 0.95);
    var buckets = 18;
    var lobeZ = [-0.12, 0.12];
    for (var i = 0; i < buckets; i++) {
      var holder = new THREE.Group();
      for (var lobe = 0; lobe < lobeZ.length; lobe++) {
        var cup = new THREE.Group();
        var shell = new THREE.Mesh(shellGeo, bucketMat);
        shell.scale.set(0.62, 0.92, 1.15);
        shell.rotation.y = -Math.PI / 2;
        var bowl = new THREE.Mesh(shellGeo, bowlMat);
        bowl.scale.set(0.42, 0.7, 0.82);
        bowl.rotation.y = -Math.PI / 2;
        bowl.position.x = 0.07;
        cup.add(shell);
        cup.add(bowl);
        cup.position.set(1.14, 0, lobeZ[lobe]);
        cup.rotation.y = -0.7;
        holder.add(cup);
      }
      var ridge = new THREE.Mesh(new THREE.BoxGeometry(0.2, 0.3, 0.03), bucketMat);
      ridge.position.set(1.1, 0, 0);
      holder.add(ridge);
      holder.rotation.z = (i / buckets) * Math.PI * 2;
      wheel.add(holder);
    }
    var rimRing = new THREE.Mesh(new THREE.TorusGeometry(1.0, 0.045, 10, 48), hubMat);
    wheel.add(rimRing);
    var spokeGeo = new THREE.BoxGeometry(0.72, 0.075, 0.075);
    for (var spokeIndex = 0; spokeIndex < 4; spokeIndex++) {
      var spoke = new THREE.Mesh(spokeGeo, hubMat);
      var spokeAngle = spokeIndex * Math.PI / 2;
      spoke.position.set(Math.cos(spokeAngle) * 0.64, Math.sin(spokeAngle) * 0.64, 0);
      spoke.rotation.z = spokeAngle;
      wheel.add(spoke);
    }
    var hubBoss = new THREE.Mesh(new THREE.CylinderGeometry(0.2, 0.2, 0.16, 28), hubMat);
    hubBoss.rotation.x = Math.PI / 2;
    wheel.add(hubBoss);
    var hubCollar = new THREE.Mesh(new THREE.TorusGeometry(0.28, 0.04, 12, 32), hubMat);
    wheel.add(hubCollar);
    var shaft = new THREE.Mesh(new THREE.CylinderGeometry(0.11, 0.11, 2.4, 24), darkMat);
    shaft.rotation.x = Math.PI / 2;
    shaft.position.z = -0.72;
    wheel.add(shaft);
    scene.add(wheel);

    var caseMat = darkMat;
    var supply = new THREE.Mesh(new THREE.CylinderGeometry(0.13, 0.13, 1.56, 16), pipeMat);
    supply.position.set(-2.55, 0.43, 0);
    scene.add(supply);
    var feed = new THREE.Mesh(new THREE.CylinderGeometry(0.13, 0.13, 0.56, 16), pipeMat);
    feed.rotation.z = Math.PI / 2;
    feed.position.set(-2.27, -0.22, 0);
    scene.add(feed);
    var spearMat = new THREE.MeshStandardMaterial({ color: 0x5e686f, metalness: 0.45, roughness: 0.38 });
    var nozzleShellMat = new THREE.MeshStandardMaterial({
      color: 0xd7dee6, metalness: 0.35, roughness: 0.4, side: THREE.DoubleSide
    });
    var nozzleAsm = new THREE.Group();
    nozzleAsm.position.set(-1.99, -0.22, 0);
    nozzleAsm.rotation.z = -Math.PI / 2;
    var spearShank = new THREE.Mesh(new THREE.CylinderGeometry(0.045, 0.045, 0.34, 16), spearMat);
    spearShank.position.y = 0.17;
    nozzleAsm.add(spearShank);
    var spearTip = new THREE.Mesh(new THREE.ConeGeometry(0.045, 0.28, 18), spearMat);
    spearTip.position.y = 0.48;
    nozzleAsm.add(spearTip);
    var nozzleShoulder = new THREE.Mesh(new THREE.CylinderGeometry(0.15, 0.13, 0.05, 24), nozzleShellMat);
    nozzleShoulder.position.y = 0.025;
    nozzleAsm.add(nozzleShoulder);
    var nozzleBarrel = new THREE.Mesh(new THREE.CylinderGeometry(0.15, 0.15, 0.32, 24), nozzleShellMat);
    nozzleBarrel.position.y = 0.21;
    nozzleAsm.add(nozzleBarrel);
    var nozzleCone = new THREE.Mesh(new THREE.CylinderGeometry(0.02, 0.15, 0.30, 24), nozzleShellMat);
    nozzleCone.position.y = 0.52;
    nozzleAsm.add(nozzleCone);
    scene.add(nozzleAsm);

    var jetColor = 0x49b7f5;
    var jetWaterMat = new THREE.MeshStandardMaterial({ color: jetColor, metalness: 0.02, roughness: 0.12, transparent: true, opacity: 0.9 });
    var beam = new THREE.Mesh(
      new THREE.CylinderGeometry(0.05, 0.08, 0.42, 12),
      jetWaterMat
    );
    beam.rotation.z = Math.PI / 2;
    beam.position.set(-1.22, -0.22, 0);
    beam.visible = water;
    scene.add(beam);

    var dropGeo = new THREE.SphereGeometry(Math.max(0.04, Math.min(0.08, dropPx * 0.0055)), 10, 8);
    var flowMat = jetWaterMat;
    var drops = [];
    var flowCount = Math.max(dropCount + 6, 12);
    for (var d = 0; d < flowCount; d++) {
      var drop = new THREE.Mesh(dropGeo, flowMat.clone());
      drop.visible = water;
      scene.add(drop);
      drops.push(drop);
    }
    var splash = new THREE.Group();
    for (var s = 0; s < 6; s++) {
      var fleck = new THREE.Mesh(new THREE.SphereGeometry(0.035, 8, 6), jetWaterMat.clone());
      var sa = (s / 6) * Math.PI * 2;
      fleck.position.set(Math.cos(sa) * 0.08, Math.sin(sa) * 0.06, ((s % 3) - 1) * 0.04);
      splash.add(fleck);
    }
    splash.position.set(-1.26, -0.24, 0);
    splash.visible = water;
    scene.add(splash);

    var deflector = new THREE.Mesh(new THREE.BoxGeometry(0.42, 0.035, 0.28), caseMat);
    deflector.position.set(-1.28, -0.5, 0);
    deflector.rotation.z = 0.55;
    scene.add(deflector);

    var tankWidth = 3.4;
    var tankDepth = 1.35;
    var tankHeight = 0.72;
    var tankX = -0.35;
    var tankBottom = -2.28;
    var glassMat = new THREE.MeshStandardMaterial({
      color: 0xd7e8f5,
      metalness: 0.05,
      roughness: 0.08,
      transparent: true,
      opacity: 0.22,
      side: THREE.DoubleSide
    });
    function tankWall(w, h, d, x, y, z) {
      var wall = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), glassMat);
      wall.position.set(x, y, z);
      scene.add(wall);
    }
    tankWall(tankWidth, 0.04, tankDepth, tankX, tankBottom, 0);
    tankWall(0.04, tankHeight, tankDepth, tankX - tankWidth / 2, tankBottom + tankHeight / 2, 0);
    tankWall(0.04, tankHeight, tankDepth, tankX + tankWidth / 2, tankBottom + tankHeight / 2, 0);
    tankWall(tankWidth, tankHeight, 0.04, tankX, tankBottom + tankHeight / 2, tankDepth / 2);
    tankWall(tankWidth, tankHeight, 0.04, tankX, tankBottom + tankHeight / 2, -tankDepth / 2);
    var tankWater = new THREE.Mesh(
      new THREE.BoxGeometry(tankWidth - 0.1, 1, tankDepth - 0.1),
      tailMat
    );
    tankWater.position.set(tankX, tankBottom, 0);
    scene.add(tankWater);
    var tankLevel = 0.12;
    var flowRate = Math.max(parseFloat(root.getAttribute("data-flow") || "0"), 0);
    var base = new THREE.Mesh(new THREE.BoxGeometry(0.7, 0.08, 0.55), hubMat);
    base.position.set(0, -1.58, -1.55);
    scene.add(base);
    var pedestal = new THREE.Mesh(new THREE.BoxGeometry(0.28, 1.28, 0.28), hubMat);
    pedestal.position.set(0, -0.9, -1.55);
    scene.add(pedestal);
    var bearing = new THREE.Mesh(new THREE.BoxGeometry(0.46, 0.46, 0.34), hubMat);
    bearing.position.set(0, 0, -1.55);
    scene.add(bearing);

    function anchor(id, x, y, z) {
      var node = new THREE.Object3D();
      node.position.set(x, y, z);
      node.userData.label = document.getElementById(id);
      scene.add(node);
      return node;
    }
    var labels = [
      anchor("pv-label-nozzle", -2.35, 1.62, 0),
      anchor("pv-label-jet", -1.55, -0.72, 0),
      anchor("pv-label-runner", 0.05, 1.9, 0),
      anchor("pv-label-shaft", 0.55, 0.42, -1.55)
    ];
    labels[3].userData.sideOnly = true;
    var tmp = new THREE.Vector3();
    var ahead = new THREE.Vector3();

    var view = window.__peltonView3d || { az: 0.62, pol: 1.16, radius: 7.4 };
    if (typeof view.zoom !== "number") view.zoom = 1;
    window.__peltonView3d = view;
    function placeCamera() {
      var aspect = Math.max(camera.aspect || 1, 0.35);
      var halfTan = Math.tan((camera.fov * Math.PI) / 360);
      var needH = (root.clientHeight || 560) < 540 ? 2.5 : 2.05;
      var fit = Math.max(7.4, 2.55 / (halfTan * aspect), needH / halfTan);
      var radius = Math.max(3.2, Math.min(14, fit * view.zoom));
      var sp = Math.sin(view.pol);
      camera.position.set(
        look.x + radius * sp * Math.sin(view.az),
        look.y + radius * Math.cos(view.pol),
        look.z + radius * sp * Math.cos(view.az)
      );
      camera.lookAt(look);
    }
    function resize() {
      var w = root.clientWidth || 640;
      var h = root.clientHeight || 560;
      root.classList.toggle("pv-compact", w < 700);
      renderer.setSize(w, h, false);
      camera.aspect = w / Math.max(h, 1);
      camera.updateProjectionMatrix();
    }
    resize();
    var observer = new ResizeObserver(resize);
    observer.observe(root);
    window.__peltonObserver = observer;

    var drag = null;
    var pointers = {};
    var pinch = null;
    function pointerCount() {
      return Object.keys(pointers).length;
    }
    root.onpointerdown = function (event) {
      if (window.__peltonToken !== token) return;
      pointers[event.pointerId] = { x: event.clientX, y: event.clientY };
      if (pointerCount() === 1) {
        drag = { id: event.pointerId, x: event.clientX, y: event.clientY, az: view.az, pol: view.pol };
        try { root.setPointerCapture(event.pointerId); } catch (error) {}
        root.style.cursor = "grabbing";
      } else {
        drag = null;
        var ids = Object.keys(pointers);
        var a = pointers[ids[0]];
        var b = pointers[ids[1]];
        pinch = { dist: Math.hypot(a.x - b.x, a.y - b.y), zoom: view.zoom };
      }
    };
    root.onpointermove = function (event) {
      if (!pointers[event.pointerId]) return;
      pointers[event.pointerId] = { x: event.clientX, y: event.clientY };
      if (pointerCount() >= 2 && pinch) {
        var ids = Object.keys(pointers);
        var a = pointers[ids[0]];
        var b = pointers[ids[1]];
        var dist = Math.hypot(a.x - b.x, a.y - b.y);
        if (pinch.dist > 8 && dist > 8) {
          view.zoom = Math.max(0.62, Math.min(1.75, pinch.zoom * (pinch.dist / dist)));
        }
        return;
      }
      if (!drag || drag.id !== event.pointerId) return;
      view.az = drag.az + (event.clientX - drag.x) * 0.008;
      view.pol = drag.pol + (event.clientY - drag.y) * 0.006;
      if (view.pol < 0.08) view.pol = 0.08;
      if (view.pol > Math.PI - 0.08) view.pol = Math.PI - 0.08;
    };
    function endDrag(event) {
      if (event && event.pointerId !== undefined) delete pointers[event.pointerId];
      drag = null;
      pinch = null;
      root.style.cursor = "grab";
    }
    root.onpointerup = endDrag;
    root.onpointercancel = endDrag;
    root.addEventListener("touchmove", function (event) {
      if (drag || pinch) event.preventDefault();
    }, { passive: false });
    root.onwheel = function (event) {
      event.preventDefault();
      var step = event.deltaY > 0 ? 1.06 : 0.94;
      view.zoom = Math.max(0.62, Math.min(1.75, view.zoom * step));
    };

    var angle = window.__peltonAngle || 0;
    var last = performance.now();
    function frame(now) {
      if (window.__peltonToken !== token) return;
      var dt = Math.min(0.05, (now - last) / 1000);
      last = now;
      if (spinning && period > 0) {
        angle = (angle + (Math.PI * 2 / period) * dt) % (Math.PI * 2);
        window.__peltonAngle = angle;
      }
      wheel.rotation.z = angle;
      var travel = Math.max(jet, 0.05);
      var rim = 1.34;
      var jetY = -0.22;
      var hitX = -1.28;
      var hit = Math.atan2(jetY, hitX);
      var sweep = Math.PI * 0.18;
      var rideTime = spinning && period > 0 ? period * (sweep / (Math.PI * 2)) : 0.12;
      var fallTime = 0.5;
      var cycle = travel + rideTime + fallTime;
      var exitA = hit + sweep;
      var exitX = Math.cos(exitA) * rim;
      var exitY = Math.sin(exitA) * rim;
      for (var n = 0; n < drops.length; n++) {
        var drop = drops[n];
        if (!water) {
          drop.visible = false;
          continue;
        }
        drop.visible = true;
        var spread = ((n % 3) - 1) * 0.05;
        var local = ((now / 1000) + (n / Math.max(drops.length, 1)) * cycle) % cycle;
        if (local < travel) {
          var jetT = local / travel;
          drop.position.set(-1.38 + jetT * (hitX - (-1.38)), jetY + spread * 0.2, spread);
          drop.material.opacity = 0.95;
        } else if (local < travel + rideTime) {
          var rideT = (local - travel) / rideTime;
          var around = hit + rideT * sweep;
          drop.position.set(Math.cos(around) * rim, Math.sin(around) * rim, spread * 0.8);
          drop.material.opacity = 0.95;
        } else {
          var fallT = (local - travel - rideTime) / fallTime;
          drop.position.set(exitX - fallT * 0.05, exitY + fallT * (-1.72 - exitY), spread);
          drop.material.opacity = fallT > 0.8 ? (1 - fallT) / 0.2 : 0.92;
        }
      }
      if (water) {
        var pulse = (Math.sin(now / (80 + travel * 90)) + 1) / 2;
        splash.visible = true;
        splash.scale.setScalar(0.65 + pulse * 0.85);
      } else {
        splash.visible = false;
      }
      var targetLevel = water ? Math.max(0.08, Math.min(0.62, 0.08 + flowRate * 0.85)) : 0.02;
      tankLevel += (targetLevel - tankLevel) * Math.min(1, dt * 1.6);
      tankWater.scale.y = tankLevel;
      tankWater.position.y = tankBottom + 0.02 + tankLevel / 2;
      placeCamera();
      var width = root.clientWidth || 1;
      var height = root.clientHeight || 1;
      for (var L = 0; L < labels.length; L++) {
        var node = labels[L];
        var el = node.userData.label;
        if (!el) continue;
        node.getWorldPosition(tmp);
        var facing = tmp.clone().sub(camera.position).dot(camera.getWorldDirection(ahead));
        var relz = camera.position.z - look.z;
        var relx = camera.position.x - look.x;
        var rely = camera.position.y - look.y;
        var span = Math.sqrt(relx * relx + rely * rely + relz * relz) || 1;
        tmp.project(camera);
        var labelX = (tmp.x * 0.5 + 0.5) * width;
        var labelY = (-tmp.y * 0.5 + 0.5) * height;
        var edge = width < 700 ? 10 : 8;
        var hintBand = width < 700 ? 46 : 34;
        var offscreen = labelX < edge || labelX > width - edge || labelY < edge || labelY > height - hintBand;
        if (facing < 0.15 || offscreen || (node.userData.sideOnly && Math.abs(relz) / span > 0.84)) {
          el.style.opacity = "0";
        } else {
          el.style.opacity = "1";
          el.style.left = labelX + "px";
          el.style.top = labelY + "px";
        }
      }
      renderer.render(scene, camera);
      window.__peltonFrame = requestAnimationFrame(frame);
    }
    window.__peltonFrame = requestAnimationFrame(frame);

    window.__peltonCleanup = function () {
      if (window.__peltonFrame) cancelAnimationFrame(window.__peltonFrame);
      window.__peltonFrame = 0;
      if (window.__peltonObserver) {
        window.__peltonObserver.disconnect();
        window.__peltonObserver = null;
      }
      scene.traverse(function (node) {
        if (node.geometry) node.geometry.dispose();
        if (node.material) {
          if (Array.isArray(node.material)) node.material.forEach(function (m) { m.dispose(); });
          else node.material.dispose();
        }
      });
      renderer.dispose();
      renderer.forceContextLoss();
      if (canvas.parentNode) canvas.parentNode.removeChild(canvas);
      if (window.__peltonRenderer === renderer) window.__peltonRenderer = null;
    };
  }

  function loadThree() {
    if (window.THREE && window.THREE.WebGLRenderer) {
      boot();
      return;
    }
    if (!window.__peltonThreePromise) {
      window.__peltonThreePromise = new Promise(function (resolve, reject) {
        var urls = [
          "https://unpkg.com/three@0.159.0/build/three.min.js",
          "https://cdn.jsdelivr.net/npm/three@0.159.0/build/three.min.js"
        ];
        var index = 0;
        function next() {
          if (index >= urls.length) {
            reject();
            return;
          }
          var loader = document.createElement("script");
          loader.src = urls[index++];
          loader.async = true;
          loader.onload = function () {
            if (window.THREE && window.THREE.WebGLRenderer) resolve();
            else next();
          };
          loader.onerror = next;
          document.head.appendChild(loader);
        }
        next();
      });
    }
    window.__peltonThreePromise.then(boot).catch(function () {
      window.__peltonThreePromise = null;
      showNote("The 3D model needs a network connection to load.");
    });
  }
  loadThree();
})();
</script>
"""


def _render_pelton_visual(data: Inputs) -> None:
    """Solid nozzle, jet, and runner under the heading. Drag orbits every angle."""
    drive = _visual_drive(data)
    bucket_count = 18
    bolt_count = 8
    drop_count = int(np.clip(round(4 + max(data.flow_m3s, 0.0) * 16), 4, 14))
    cups = "".join('<span class="cup">&#8203;</span>' for _ in range(bucket_count))
    bolts = "".join('<span class="bolt">&#8203;</span>' for _ in range(bolt_count))
    drops = "".join('<span class="drop">&#8203;</span>' for _ in range(drop_count))
    cup_rules = "\n".join(
        ".pv-wheel > .cup:nth-child("
        f"{index}) {{ transform: rotateZ({(index - 1) * (360 / bucket_count):.1f}deg) translateX(96px); }}"
        for index in range(1, bucket_count + 1)
    )
    bolt_rules = "\n".join(
        ".pv-disk > .bolt:nth-child("
        f"{index}) {{ transform: rotateZ({(index - 1) * (360 / bolt_count):.1f}deg) translateX(58px); }}"
        for index in range(1, bolt_count + 1)
    )
    drop_px = float(drive["drop_px"])
    half_drop = drop_px / 2.0
    jet_seconds = float(drive["jet_seconds"]) or 0.8
    period = float(drive["period"])
    spin_css = (
        f"animation: pv-spin {period:.2f}s linear infinite;"
        if drive["spinning"]
        else "animation: none;"
    )
    water_css = "display: block;" if drive["has_water"] else "display: none;"
    state = "Jet is turning the runner" if drive["spinning"] else "Wheel stopped"
    markup = dedent(
        f"""
        <style>
        .pv-card {{ margin: 0 0 0.75rem 0; }}
        .pv-readout {{
            margin: 0 0 0.4rem 0;
            color: #f7fbff;
            font-size: 0.95rem;
            line-height: 1.45;
            overflow-wrap: anywhere;
            background: linear-gradient(115deg, #06283d 0%, #0b4f8a 100%);
            border-radius: 12px;
            padding: 0.55rem 0.85rem;
        }}
        .pv-readout strong {{ font-size: 1.12rem; }}
        .pv-state {{ margin-left: 0.5rem; color: #d7ebff; font-size: 0.85rem; }}
        .pv-viewport {{
            position: relative;
            height: 560px;
            max-width: 100%;
            perspective: 1300px;
            overflow: hidden;
            border-radius: 16px;
            cursor: grab;
            touch-action: none;
            user-select: none;
            -webkit-user-select: none;
            background: #ffffff;
            border: 1px solid rgba(20, 40, 70, 0.12);
        }}
        .pv-viewport canvas {{
            position: absolute;
            inset: 0;
            width: 100%;
            height: 100%;
            display: block;
            z-index: 1;
        }}
        .pv-label {{
            position: absolute;
            z-index: 2;
            margin: 0;
            color: #1a1a1a;
            font-size: 13px;
            line-height: 1;
            white-space: nowrap;
            font-family: Georgia, "Times New Roman", serif;
            transform: translate(-50%, -130%);
            pointer-events: none;
        }}
        .pv-note {{
            position: absolute;
            inset: 0;
            z-index: 2;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0;
            padding: 1rem;
            text-align: center;
            color: #334455;
        }}
        .pv-world {{
            position: absolute;
            left: calc(50% + 36px);
            top: 46%;
            width: 0;
            height: 0;
            transform-style: preserve-3d;
        }}
        .pv-part {{
            position: absolute;
            left: 0;
            top: 0;
            transform-style: preserve-3d;
        }}
        .pv-casing {{
            width: 268px;
            height: 188px;
            margin: -176px 0 0 -134px;
            border: 4px solid #1c1c1c;
            border-bottom: none;
            border-radius: 134px 134px 0 0;
            background: rgba(255, 255, 255, 0.2);
            transform: translateZ(-30px);
        }}
        .pv-casing-front {{
            width: 268px;
            height: 188px;
            margin: -176px 0 0 -134px;
            border: 4px solid #1c1c1c;
            border-bottom: none;
            border-radius: 134px 134px 0 0;
            transform: translateZ(26px);
            pointer-events: none;
        }}
        .pv-wheel {{
            width: 0;
            height: 0;
            {spin_css}
        }}
        .pv-wheel > .cup, .pv-disk > .bolt {{
            position: absolute;
        }}
        .pv-wheel > .cup {{
            width: 28px;
            height: 42px;
            margin: -21px 0 0 -8px;
            background: #ffffff;
            border: 2.5px solid #1a56b0;
            border-radius: 42% 42% 48% 48%;
            box-shadow: inset 8px 0 0 rgba(26, 86, 176, 0.18);
        }}
        .pv-disk {{
            width: 86px;
            height: 86px;
            margin: -43px 0 0 -43px;
            border-radius: 50%;
            background: #ffffff;
            border: 3px solid #1a56b0;
            transform: translateZ(8px);
        }}
        .pv-disk > .bolt {{
            left: 50%;
            top: 50%;
            width: 11px;
            height: 11px;
            margin: -5.5px 0 0 -5.5px;
            border-radius: 50%;
            background: #ffffff;
            border: 2px solid #1a56b0;
        }}
        .pv-hub {{
            width: 34px;
            height: 34px;
            margin: -17px 0 0 -17px;
            border-radius: 50%;
            background: #ffffff;
            border: 3px solid #1a56b0;
            box-shadow: inset 0 0 0 6px #ffffff, inset 0 0 0 9px #1a56b0;
            transform: translateZ(16px);
        }}
        .pv-shaft {{
            width: 78px;
            height: 6px;
            margin-top: -3px;
            background: #1c1c1c;
            border-radius: 3px;
            transform: translate3d(70px, 0, 4px);
        }}
        .pv-inlet {{
            width: 16px;
            height: 42px;
            margin: -21px 0 0 -8px;
            border: 3px solid #1c1c1c;
            border-bottom: none;
            background: linear-gradient(#d7f2ff, #5eb7e8);
            transform: translate3d(-214px, -108px, 0);
        }}
        .pv-bend {{
            width: 58px;
            height: 58px;
            border-left: 14px solid #7ec8ea;
            border-bottom: 14px solid #7ec8ea;
            border-radius: 0 0 0 46px;
            box-shadow: -3px 0 0 #1c1c1c, 0 3px 0 #1c1c1c;
            transform: translate3d(-206px, -72px, 0);
        }}
        .pv-nozzle-body {{
            width: 46px;
            height: 22px;
            margin-top: -11px;
            background: #ffffff;
            border: 3px solid #1c1c1c;
            border-radius: 2px 12px 12px 2px;
            transform: translate3d(-162px, 0, 0);
        }}
        .pv-spear {{
            width: 0;
            height: 0;
            border-top: 5px solid transparent;
            border-bottom: 5px solid transparent;
            border-left: 22px solid #1c1c1c;
            transform: translate3d(-176px, -5px, 4px);
        }}
        .pv-jet-beam {{
            width: 62px;
            height: {drop_px:.0f}px;
            margin-top: -{half_drop:.0f}px;
            border-radius: 8px;
            {water_css}
            opacity: {float(drive["beam_opacity"]):.2f};
            background: repeating-linear-gradient(90deg, #2f86d6 0 10px, #d7f1ff 10px 16px);
            transform: translate3d(-158px, 0, 2px);
            animation: pv-flow {jet_seconds:.2f}s linear infinite;
        }}
        .drop {{
            position: absolute;
            width: {drop_px:.0f}px;
            height: {drop_px:.0f}px;
            margin: -{half_drop:.1f}px 0 0 -{half_drop:.1f}px;
            border-radius: 50%;
            {water_css}
            background: radial-gradient(circle at 32% 32%, #ffffff, #4aa3e8 55%, #1a56b0);
            border: 1px solid #1a56b0;
            animation: pv-jet {jet_seconds:.2f}s linear infinite;
        }}
        .splash {{
            width: 26px;
            height: 26px;
            margin: -13px 0 0 -13px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(255,255,255,0.95), rgba(47, 134, 214, 0.15) 70%);
            {water_css}
        }}
        .pv-deflector {{
            width: 34px;
            height: 4px;
            background: #1c1c1c;
            transform: translate3d(-124px, 22px, 2px) rotateZ(28deg);
        }}
        .pv-tail {{
            width: 300px;
            height: 42px;
            margin-left: -150px;
            border-top: 3px solid #1a56b0;
            background:
                repeating-linear-gradient(90deg, transparent 0 18px, rgba(26, 86, 176, 0.25) 18px 28px),
                linear-gradient(#8ec9f2, #d7f1ff);
            transform: translate3d(0, 128px, -12px);
        }}
        .pv-tag {{
            color: #1a1a1a;
            font-size: 12px;
            line-height: 1;
            white-space: nowrap;
            font-family: Georgia, "Times New Roman", serif;
        }}
        .pv-tag-nozzle {{ transform: translate3d(-262px, -124px, 12px); }}
        .pv-tag-jet {{ transform: translate3d(-168px, 46px, 12px); }}
        .pv-tag-runner {{ transform: translate3d(-18px, -132px, 12px); }}
        .pv-tag-shaft {{ transform: translate3d(86px, -22px, 12px); }}
        .pv-hint {{
            position: absolute;
            left: 0;
            right: 0;
            bottom: 8px;
            margin: 0;
            padding: 0 10px;
            text-align: center;
            color: #3e4c59;
            font-size: 0.84rem;
            line-height: 1.3;
            pointer-events: none;
            z-index: 3;
        }}
        .pv-compact .pv-label {{ font-size: 11px; }}
        .pv-compact .pv-hint {{ font-size: 0.75rem; }}
        @media (max-width: 700px) {{
            .pv-readout {{ font-size: 0.88rem; }}
            .pv-readout strong {{ font-size: 1rem; }}
            .pv-state {{ display: block; margin: 0.15rem 0 0 0; }}
            .pv-viewport {{
                height: min(70vh, 520px);
                min-height: 340px;
                border-radius: 12px;
            }}
            .hero {{
                flex-wrap: wrap;
                gap: 0.7rem;
                padding: 0.85rem 0.9rem;
            }}
            .hero h1 {{ font-size: 1.35rem; }}
            .hero > div {{ min-width: 0; flex: 1 1 12rem; }}
            section.main > div.block-container {{
                padding-left: 0.75rem;
                padding-right: 0.75rem;
            }}
            div[data-testid="stHorizontalBlock"] {{
                flex-wrap: wrap !important;
            }}
            div[data-testid="stHorizontalBlock"] > div {{
                flex: 1 1 46% !important;
                min-width: 46% !important;
                width: auto !important;
            }}
            div[data-testid="stHorizontalBlock"]:has(.katex) {{
                flex-direction: column !important;
            }}
            div[data-testid="stHorizontalBlock"]:has(.katex) > div {{
                flex: 1 1 100% !important;
                min-width: 100% !important;
                width: 100% !important;
            }}
            div[data-testid="stHorizontalBlock"]:has(.katex) > div + div {{
                margin-top: 0.2rem;
            }}
            .katex {{
                font-size: 0.86em !important;
            }}
            .katex-display {{
                margin: 0.3em 0 !important;
                overflow-x: auto;
                overflow-y: hidden;
            }}
            div[data-testid="stMetric"] {{
                padding: 0.4rem 0.45rem 0.2rem 0.45rem !important;
            }}
            div[data-testid="stMetricLabel"],
            div[data-testid="stMetricLabel"] * {{
                font-size: 0.74rem !important;
                line-height: 1.2 !important;
                white-space: normal !important;
            }}
            div[data-testid="stMetricValue"],
            div[data-testid="stMetricValue"] * {{
                font-size: 1.02rem !important;
                line-height: 1.15 !important;
                overflow: visible !important;
                text-overflow: clip !important;
                white-space: nowrap !important;
            }}
            div[data-testid="stCaptionContainer"] {{
                font-size: 0.8rem !important;
                line-height: 1.35 !important;
                white-space: normal !important;
            }}
            [data-testid="stElementContainer"]:has(#symbol-guide-mark) + [data-testid="stElementContainer"] [data-testid="stMarkdown"],
            [data-testid="stElementContainer"]:has(#symbol-guide-mark) + [data-testid="stElementContainer"] table {{
                width: 100% !important;
                max-width: 100% !important;
                min-width: 0 !important;
                overflow: visible !important;
            }}
            [data-testid="stElementContainer"]:has(#symbol-guide-mark) + [data-testid="stElementContainer"] table,
            [data-testid="stElementContainer"]:has(#symbol-guide-mark) + [data-testid="stElementContainer"] tbody {{
                display: block !important;
                width: 100% !important;
            }}
            [data-testid="stElementContainer"]:has(#symbol-guide-mark) + [data-testid="stElementContainer"] thead {{
                display: none !important;
            }}
            [data-testid="stElementContainer"]:has(#symbol-guide-mark) + [data-testid="stElementContainer"] tr {{
                display: block !important;
                width: auto !important;
                margin: 0 0 0.7rem 0;
                padding: 0.7rem 0.75rem 0.45rem 0.75rem;
                border: 1px solid rgba(120, 156, 186, 0.45);
                border-radius: 12px;
            }}
            [data-testid="stElementContainer"]:has(#symbol-guide-mark) + [data-testid="stElementContainer"] td {{
                display: block !important;
                width: 100% !important;
                max-width: 100% !important;
                box-sizing: border-box;
                text-align: left !important;
                white-space: normal !important;
                overflow: visible !important;
                overflow-wrap: anywhere;
                border: none !important;
                padding: 0.15rem 0 0.4rem 0 !important;
                font-size: 0.86rem;
                line-height: 1.4;
            }}
            [data-testid="stElementContainer"]:has(#symbol-guide-mark) + [data-testid="stElementContainer"] td::before {{
                content: "";
                display: block;
                margin-bottom: 0.08rem;
                font-size: 0.7rem;
                font-weight: 650;
                letter-spacing: 0.01em;
                opacity: 0.72;
            }}
            [data-testid="stElementContainer"]:has(#symbol-guide-mark) + [data-testid="stElementContainer"] td:nth-child(1) {{
                font-size: 1.05rem;
                font-weight: 700;
                padding-bottom: 0.45rem !important;
                margin-bottom: 0.15rem;
                border-bottom: 1px solid rgba(120, 156, 186, 0.35) !important;
            }}
            [data-testid="stElementContainer"]:has(#symbol-guide-mark) + [data-testid="stElementContainer"] td:nth-child(1)::before {{ content: "Symbol"; }}
            [data-testid="stElementContainer"]:has(#symbol-guide-mark) + [data-testid="stElementContainer"] td:nth-child(2)::before {{ content: "What it represents"; }}
            [data-testid="stElementContainer"]:has(#symbol-guide-mark) + [data-testid="stElementContainer"] td:nth-child(3)::before {{ content: "Meaning on the Pelton wheel"; }}
            [data-testid="stElementContainer"]:has(#symbol-guide-mark) + [data-testid="stElementContainer"] td:nth-child(4)::before {{ content: "Unit"; }}
            [data-testid="stElementContainer"]:has(#symbol-guide-mark) + [data-testid="stElementContainer"] td:nth-child(5)::before {{ content: "How it is used"; }}
        }}
        {cup_rules}
        {bolt_rules}
        .drop:nth-child(1) {{ animation-delay: 0s; }}
        .drop:nth-child(2) {{ animation-delay: -0.12s; }}
        .drop:nth-child(3) {{ animation-delay: -0.24s; }}
        .drop:nth-child(4) {{ animation-delay: -0.36s; }}
        .drop:nth-child(5) {{ animation-delay: -0.48s; }}
        .drop:nth-child(6) {{ animation-delay: -0.08s; }}
        .drop:nth-child(7) {{ animation-delay: -0.2s; }}
        .drop:nth-child(8) {{ animation-delay: -0.32s; }}
        @keyframes pv-spin {{
            from {{ transform: rotateZ(0deg); }}
            to {{ transform: rotateZ(360deg); }}
        }}
        @keyframes pv-flow {{
            from {{ background-position: 0 0; }}
            to {{ background-position: 44px 0; }}
        }}
        @keyframes pv-jet {{
            from {{ transform: translate3d(-158px, -2px, 0); opacity: 1; }}
            88% {{ opacity: 1; }}
            to {{ transform: translate3d(-98px, -2px, 0); opacity: 0.15; }}
        }}
        @media (prefers-reduced-motion: reduce) {{
            .pv-wheel, .drop, .pv-jet-beam {{ animation: none; }}
        }}
        </style>
        <div class="pv-card">
        <p class="pv-readout">Water force <strong>{drive["force_text"]}</strong> · Runner <strong>{drive["rpm_text"]} rpm</strong> <span class="pv-state">{state}</span></p>
        <div class="pv-viewport" id="pv-viewport" data-period="{period:.3f}" data-jet="{jet_seconds:.3f}" data-spinning="{"1" if drive["spinning"] else "0"}" data-water="{"1" if drive["has_water"] else "0"}" data-drops="{drop_count}" data-drop="{drop_px:.1f}" data-flow="{data.flow_m3s:.4f}">
        <span class="pv-label" id="pv-label-nozzle">Nozzle</span>
        <span class="pv-label" id="pv-label-jet">Jet of water</span>
        <span class="pv-label" id="pv-label-runner">Runner</span>
        <span class="pv-label" id="pv-label-shaft">Shaft</span>
        <p class="pv-hint">Drag or swipe to turn the turbine. Scroll or pinch to zoom.</p>
        </div>
        </div>
        """
    ).strip()
    st.html(markup + _PELTON_SCRIPT, unsafe_allow_javascript=True)


def _render_header(spin_seconds: float, data: Inputs) -> None:
    # Keep every line flush left. Indented HTML is rendered as a code block.
    markup = dedent(
        f"""
        <style>
        .hero {{
            display: flex;
            gap: 1.1rem;
            align-items: center;
            background: linear-gradient(115deg, #06283d 0%, #0b4f8a 55%, #1a7abf 100%);
            color: #f7fbff;
            border-radius: 16px;
            padding: 1rem 1.2rem;
            margin-bottom: 0.8rem;
        }}
        .hero h1 {{
            margin: 0;
            font-size: 1.7rem;
            line-height: 1.2;
            color: #ffffff;
        }}
        .hero p {{
            margin: 0.25rem 0 0 0;
            color: #e5f2ff;
        }}
        .group-pill {{
            display: inline-block;
            margin-top: 0.55rem;
            padding: 0.15rem 0.6rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.16);
            font-weight: 600;
        }}
        .turbine {{
            position: relative;
            width: 148px;
            height: 112px;
            flex: 0 0 auto;
        }}
        .jet {{
            position: absolute;
            left: 0;
            top: 52px;
            width: 52px;
            height: 8px;
            border-radius: 8px;
            background: repeating-linear-gradient(90deg, #d7f4ff 0 8px, rgba(215, 244, 255, 0.15) 8px 16px);
            animation: jetflow 0.55s linear infinite;
        }}
        .nozzle {{
            position: absolute;
            left: 48px;
            top: 48px;
            border-top: 8px solid transparent;
            border-bottom: 8px solid transparent;
            border-left: 12px solid #d7f4ff;
        }}
        .wheel {{
            position: absolute;
            left: 62px;
            top: 14px;
            width: 84px;
            height: 84px;
            border: 3px solid #f4fbff;
            border-radius: 50%;
            box-shadow: inset 0 0 0 12px rgba(159, 208, 245, 0.35);
            animation: spin {spin_seconds:.2f}s linear infinite;
        }}
        .wheel::after {{
            content: "";
            position: absolute;
            left: 50%;
            top: 50%;
            width: 14px;
            height: 14px;
            margin: -7px 0 0 -7px;
            border-radius: 50%;
            background: #7ec8ff;
            border: 2px solid #08304f;
        }}
        .wheel span {{
            position: absolute;
            left: 50%;
            top: 50%;
            width: 12px;
            height: 12px;
            margin: -6px 0 0 -6px;
            border-radius: 50%;
            background: #e7f3ff;
            border: 1px solid #08304f;
        }}
        .wheel span:nth-child(1) {{ transform: rotate(0deg) translateX(32px); }}
        .wheel span:nth-child(2) {{ transform: rotate(36deg) translateX(32px); }}
        .wheel span:nth-child(3) {{ transform: rotate(72deg) translateX(32px); }}
        .wheel span:nth-child(4) {{ transform: rotate(108deg) translateX(32px); }}
        .wheel span:nth-child(5) {{ transform: rotate(144deg) translateX(32px); }}
        .wheel span:nth-child(6) {{ transform: rotate(180deg) translateX(32px); }}
        .wheel span:nth-child(7) {{ transform: rotate(216deg) translateX(32px); }}
        .wheel span:nth-child(8) {{ transform: rotate(252deg) translateX(32px); }}
        .wheel span:nth-child(9) {{ transform: rotate(288deg) translateX(32px); }}
        .wheel span:nth-child(10) {{ transform: rotate(324deg) translateX(32px); }}
        div[data-testid="stMetric"] {{
            background: rgba(11, 94, 168, 0.08);
            border: 1px solid rgba(11, 94, 168, 0.22);
            border-radius: 12px;
            padding: 0.65rem 0.75rem 0.35rem 0.75rem;
            animation: rise 0.65s ease both;
        }}
        @keyframes spin {{
            from {{ transform: rotate(0deg); }}
            to {{ transform: rotate(360deg); }}
        }}
        @keyframes jetflow {{
            from {{ background-position: 0 0; }}
            to {{ background-position: 16px 0; }}
        }}
        @keyframes rise {{
            from {{ opacity: 0; transform: translateY(8px); }}
            to {{ opacity: 1; transform: none; }}
        }}
        @media (prefers-reduced-motion: reduce) {{
            .wheel, .jet, div[data-testid="stMetric"] {{
                animation: none;
            }}
        }}
        </style>
        <div class="hero">
        {_wheel_markup()}
        <div>
        <h1>Pelton Wheel Turbine</h1>
        <p>Basic Sizing &amp; Power Estimator</p>
        <span class="group-pill">Group {GROUP_NUMBER}</span>
        </div>
        </div>
        """
    ).strip()
    st.html(markup)
    _render_pelton_visual(data)
    st.caption("The wheel and jet are an illustration. Rotation speed is only a visual cue, not a scale model.")


def _render_team() -> None:
    rows = "\n".join(f"| {name} | {enrollment} |" for name, enrollment in MEMBERS)
    st.markdown(
        f"""
**Group {GROUP_NUMBER}**

| Member | Enrollment number |
| --- | --- |
{rows}
"""
    )


def _render_sidebar() -> Inputs:
    st.sidebar.header("Inputs")
    st.sidebar.caption("Site data for one Pelton operating point.")

    gross_head_m = st.sidebar.number_input(
        "Gross head (m)",
        min_value=-500.0,
        max_value=3000.0,
        value=265.0,
        step=1.0,
        format="%.2f",
        help="Vertical distance from the free water surface to the nozzle centreline.",
    )
    penstock_loss_m = st.sidebar.number_input(
        "Penstock loss (m)",
        min_value=-100.0,
        max_value=3000.0,
        value=15.0,
        step=0.5,
        format="%.2f",
        help="Head lost in the penstock, in metres of water.",
    )
    flow_per_rpm = 0.35 / 600.0
    if "flow_m3s" not in st.session_state:
        st.session_state.runner_speed_rpm = 100.0
        st.session_state.flow_m3s = 100.0 * flow_per_rpm

    def _sync_speed_from_flow() -> None:
        speed = float(st.session_state.flow_m3s) / flow_per_rpm
        speed = min(6000.0, max(-500.0, speed))
        st.session_state.runner_speed_rpm = speed
        st.session_state.flow_m3s = speed * flow_per_rpm

    def _sync_flow_from_speed() -> None:
        flow = float(st.session_state.runner_speed_rpm) * flow_per_rpm
        flow = min(30.0, max(-1.0, flow))
        st.session_state.flow_m3s = flow
        st.session_state.runner_speed_rpm = flow / flow_per_rpm

    flow_m3s = st.sidebar.number_input(
        "Water flow rate (m³/s)",
        min_value=-1.0,
        max_value=30.0,
        step=0.01,
        format="%.3f",
        key="flow_m3s",
        on_change=_sync_speed_from_flow,
        help="Total discharge through all jets. Changing this updates runner speed.",
    )
    runner_speed_rpm = st.sidebar.number_input(
        "Runner speed (rpm)",
        min_value=-500.0,
        max_value=6000.0,
        step=5.0,
        format="%.1f",
        key="runner_speed_rpm",
        on_change=_sync_flow_from_speed,
        help="Rotational speed of the runner. Starts at 100 rpm. Changing this updates flow rate.",
    )

    st.sidebar.divider()
    st.sidebar.subheader("Design coefficients")
    velocity_coefficient = st.sidebar.slider(
        "Velocity coefficient Cv",
        min_value=0.90,
        max_value=0.99,
        value=0.98,
        step=0.01,
        help="Actual jet speed divided by the ideal speed from the net head.",
    )
    speed_ratio = st.sidebar.slider(
        "Speed ratio φ",
        min_value=0.40,
        max_value=0.50,
        value=0.46,
        step=0.01,
        help="Bucket speed divided by jet speed. About 0.46 is a common design value.",
    )
    jet_count = st.sidebar.slider(
        "Number of jets",
        min_value=1,
        max_value=6,
        value=1,
        step=1,
        help="Equal jets sharing the total flow. Used for the jet diameter.",
    )
    st.sidebar.caption("Group 25 · Pelton wheel estimator")
    return Inputs(
        gross_head_m=float(gross_head_m),
        penstock_loss_m=float(penstock_loss_m),
        flow_m3s=float(flow_m3s),
        runner_speed_rpm=float(runner_speed_rpm),
        velocity_coefficient=float(velocity_coefficient),
        speed_ratio=float(speed_ratio),
        jet_count=int(jet_count),
    )


def _render_formulas(data: Inputs, results: Results) -> None:
    st.subheader("Formulas")
    st.markdown(
        "Fresh water is taken as 1000 kg/m³ and g as 9.81 m/s². "
        "Bucket speed is the design peripheral speed, set by the speed ratio. "
        "Runner speed then fixes the pitch-circle diameter that produces that bucket speed."
    )
    left, right = st.columns(2)
    with left:
        st.markdown("**Net head**")
        st.latex(r"H_{net} = H_g - h_f")
        st.latex(
            rf"H_{{net}} = {data.gross_head_m:.2f} - {data.penstock_loss_m:.2f} "
            rf"= {results.net_head_m:.2f}\ \mathrm{{m}}"
        )
        st.markdown("**Bucket speed**")
        st.latex(r"u = \phi \, V_j")
        st.latex(
            rf"u = {data.speed_ratio:.2f} \times {results.jet_velocity_ms:.2f} "
            rf"= {results.bucket_speed_ms:.2f}\ \mathrm{{m/s}}"
        )
        st.markdown("**Pitch diameter**")
        st.latex(r"D = \dfrac{60 \, u}{\pi N}")
        st.latex(
            rf"D = {results.pitch_diameter_m:.3f}\ \mathrm{{m}}"
        )
    with right:
        st.markdown("**Water jet velocity**")
        st.latex(r"V_j = C_v \sqrt{2 g H_{net}}")
        st.latex(
            rf"V_j = {data.velocity_coefficient:.2f}"
            rf"\sqrt{{2 \times 9.81 \times {results.net_head_m:.2f}}}"
            rf"= {results.jet_velocity_ms:.2f}\ \mathrm{{m/s}}"
        )
        st.markdown("**Hydraulic power**")
        st.latex(r"P = \dfrac{\rho g Q H_{net}}{1000}")
        st.latex(
            rf"P = {results.hydraulic_power_kw:.2f}\ \mathrm{{kW}}"
        )
        st.markdown("**Jet diameter**")
        st.latex(r"d = \sqrt{\dfrac{4Q}{z \pi V_j}}")
        st.latex(
            rf"d = {results.jet_diameter_m * 1000:.1f}\ \mathrm{{mm}}"
        )
    st.markdown(
        '<p id="symbol-guide-mark"><strong>Symbols in these formulas</strong></p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
| Symbol | What it represents | Meaning on the Pelton wheel | Unit | How it is used |
| --- | --- | --- | --- | --- |
| H_net | Net head | Head still available at the nozzle after the penstock has taken its losses. This is the head that actually drives the jet. | m | H_net = H_g − h_f. It is then used to find jet velocity and hydraulic power. |
| H_g | Gross head | Vertical distance from the free water surface in the reservoir down to the nozzle centreline, before any pipe loss. | m | The starting head. Penstock loss is subtracted from it. |
| h_f | Penstock head loss | Head used up by friction and fittings as water travels down the penstock. | m | Subtracted from gross head so the jet is not credited with head it never receives. |
| V_j | Jet velocity | Actual speed of the water jet as it leaves the nozzle and strikes the buckets. | m/s | V_j = C_v √(2 g H_net). Bucket speed and jet diameter are both based on this speed. |
| C_v | Velocity coefficient | Share of the ideal nozzle speed that the real jet reaches. A value near 0.98 means a small loss inside the nozzle. | — | Multiplies the ideal speed √(2 g H_net) to give the real jet speed. |
| g | Gravitational acceleration | The pull of gravity that converts a height of water into speed. Taken here as 9.81. | m/s² | Appears inside √(2 g H_net), which is the ideal jet speed from the net head. |
| u | Bucket speed | Design speed of the bucket at the pitch circle, in the direction of the jet. Sometimes written U. | m/s | u = φ V_j. That peripheral speed is then used to size the pitch diameter. |
| φ | Speed ratio | How fast the bucket is meant to run compared with the jet. About 0.46 is a usual Pelton design value. | — | Sets bucket speed as a fraction of jet velocity. |
| D | Pitch diameter | Diameter of the circle through the centre of the jet where it meets the buckets. | m | D = 60 u / (π N). A faster wheel, at the same bucket speed, needs a smaller pitch circle. |
| N | Runner speed | Rotational speed of the wheel and shaft. | rpm | With bucket speed, it fixes the pitch diameter. |
| π | Pi | The circle constant, about 3.1416. It links a diameter to the distance a point on the rim travels in one revolution. | — | Turns revolutions per minute into the peripheral speed used to solve for D. |
| P | Hydraulic power | Power available from the water supplied at the net head. This is the water power, before mechanical losses in the wheel. | kW | P = ρ g Q H_net / 1000. |
| ρ | Density of water | Mass of water in one cubic metre. Fresh water is taken as 1000. | kg/m³ | Converts flow and head into a mass-flow power. |
| Q | Flow rate | Total discharge delivered to the turbine, shared by all jets. | m³/s | Multiplies head in the power formula, and is split among the jets when the jet diameter is found. |
| 1000 | Kilowatt conversion | Number of watts in one kilowatt. It is not a site measurement. | W/kW | Divides ρ g Q H_net, which is in watts, so the reported power is in kilowatts. |
| d | Jet diameter | Diameter of the opening of one nozzle. The formula gives metres; the page also shows it in millimetres. | m | d = √(4Q / (z π V_j)), from the area needed to pass each jet’s share of the flow at speed V_j. |
| z | Number of jets | How many equal nozzles share the total flow. | — | Each jet carries Q/z, so more jets make each jet, and d, smaller. |
"""
    )


def _render_results(data: Inputs, results: Results) -> None:
    st.subheader("Calculated outputs")
    loss_share = 100.0 * data.penstock_loss_m / data.gross_head_m
    st.caption(
        f"Flow used: {data.flow_m3s:.3f} m³/s ({data.flow_m3s * 1000:.1f} L/s). "
        f"Penstock loss is {loss_share:.1f}% of gross head."
    )
    net, jet, bucket, power = st.columns(4)
    net.metric("Net head", f"{results.net_head_m:.2f} m")
    jet.metric("Jet velocity", f"{results.jet_velocity_ms:.2f} m/s")
    bucket.metric("Bucket speed", f"{results.bucket_speed_ms:.2f} m/s")
    power.metric("Hydraulic power", f"{results.hydraulic_power_kw:,.2f} kW")

    st.subheader("Basic sizing")
    pitch, jet_d, ratio, ns = st.columns(4)
    pitch.metric("Pitch diameter", f"{results.pitch_diameter_m:.3f} m")
    jet_d.metric("Jet diameter", f"{results.jet_diameter_m * 1000:.1f} mm")
    ratio.metric("Jet ratio D/d", f"{results.jet_ratio:.2f}")
    ns.metric("Specific speed Ns", f"{results.specific_speed:.1f}")
    st.caption(
        "Ns = N √P / H^1.25 with N in rpm, hydraulic power P in kW, and net head H in m. "
        f"Pitch diameter is also {results.pitch_diameter_m * 1000:.0f} mm."
    )


def main() -> None:
    st.set_page_config(
        page_title="Pelton Wheel Turbine Estimator",
        page_icon="💧",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    plt.rcParams.update(
        {
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 8.5,
            "figure.facecolor": "white",
        }
    )

    data = _render_sidebar()
    _render_header(_spin_seconds(data.runner_speed_rpm), data)
    _render_team()

    errors, warnings = validate(data)
    for message in errors:
        st.error(message)
    if errors:
        st.stop()

    for message in warnings:
        st.warning(message)
    if not warnings:
        st.success("Inputs are inside the usual range for a basic Pelton estimate.")

    results = evaluate(data)
    _render_formulas(data, results)
    _render_results(data, results)

    st.subheader("Charts")
    figure = build_figure(data, results)
    st.pyplot(figure, width="stretch")
    plt.close(figure)
    st.caption(
        "Both charts use the sidebar coefficients. Gridlines are on, and the markers show "
        "the current operating point. Power lines at nearby heads show how sensitive the "
        "estimate is to net head."
    )
    st.divider()
    st.caption(
        f"Group {GROUP_NUMBER} · Pelton Wheel Turbine Basic Sizing & Power Estimator"
    )


if __name__ == "__main__":
    main()
