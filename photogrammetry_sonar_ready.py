"""
photogrammetry_sonar_ready.py
---------------------------------
Archivo generado para prácticas de análisis estático (SonarQube).
Es un programa de ejemplo de fotogrametría en Python, diseñado para tener
estructura modular, funciones, clases, docstrings y comentarios que ejercicios
comunes de herramientas de análisis estático detectarán y podrán evaluar.

Este archivo NO pretende ser una implementación de producción de algoritmos
fotogramétricos completos. Muchos cálculos están simplificados o simulados,
pero el flujo y la arquitectura son representativos de una aplicación real:
- lectura de imágenes
- extracción de características
- emparejamiento de features
- estimación de cámara (pose)
- aerotriangulación simplificada
- generación de ortomosaico (simulado)
- creación de reportes y pruebas unitarias

Incluye: typing, logging, manejo de errores, validaciones y CLI.

Nota: el código es intencionalmente largo para fines de análisis estático.
"""

# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import math
import os
import random
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Configuración de logging y utilidades
# ---------------------------------------------------------------------------

LOG_LEVEL = logging.DEBUG

logger = logging.getLogger("photogrammetry_sonar_ready")
logger.setLevel(LOG_LEVEL)
_handler = logging.StreamHandler(sys.stdout)
_handler.setLevel(LOG_LEVEL)
_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
_handler.setFormatter(_formatter)
logger.addHandler(_handler)


def ensure_dir(path: str) -> str:
    """Ensure a directory exists and return its absolute path.

    Args:
        path: Path to directory.

    Returns:
        Absolute path to the directory.
    """
    p = Path(path)
    if p.exists() and not p.is_dir():
        raise NotADirectoryError(f"Path exists and is not a directory: {path}")
    p.mkdir(parents=True, exist_ok=True)
    abs_path = str(p.resolve())
    logger.debug("ensure_dir: %s", abs_path)
    return abs_path


# ---------------------------------------------------------------------------
# Tipos y estructuras de datos simples
# ---------------------------------------------------------------------------

@dataclass
class ImageMeta:
    """Metadata for a single image (simulated)."""

    id: str
    file_path: str
    gps: Optional[Tuple[float, float, float]] = None
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    camera_matrix: Optional[List[List[float]]] = None


@dataclass
class Feature:
    """Represents a detected feature in an image.

    x: float
        x coordinate in pixels
    y: float
        y coordinate in pixels
    descriptor: Tuple[int, ...]
        A simple numeric descriptor for matching
    """

    x: float
    y: float
    descriptor: Tuple[int, ...]
    octave: int = 0


@dataclass
class Match:
    """Represents a match between two features in different images."""

    image_id_a: str
    image_id_b: str
    feat_idx_a: int
    feat_idx_b: int
    score: float


# ---------------------------------------------------------------------------
# Módulo de lectura y simulación de imágenes
# ---------------------------------------------------------------------------

class ImageReader:
    """Clase simulada para leer imágenes desde un directorio.

    Esta clase no procesa imágenes reales. Genera metadatos y 'pixeles'
    aleatorios para permitir que el pipeline funcione sin librerías externas.
    """

    def __init__(self, directory: str) -> None:
        self.directory = ensure_dir(directory)
        logger.info("ImageReader initialized for %s", self.directory)

    def list_images(self) -> List[ImageMeta]:
        """List images in the directory and produce ImageMeta objects.

        The function accepts common image extensions. For files that are not
        images, it will ignore them. GPS values are simulated for missing data.
        """
        img_ext = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
        images: List[ImageMeta] = []
        p = Path(self.directory)
        files = [f for f in p.iterdir() if f.suffix.lower() in img_ext]
        files.sort()
        for idx, f in enumerate(files):
            gps = (random.uniform(-76.7, -76.5), random.uniform(4.5, 4.9), 100.0 + idx)
            meta = ImageMeta(
                id=f.stem,
                file_path=str(f.resolve()),
                gps=gps,
                yaw=random.uniform(0, 360),
                pitch=random.uniform(-10, 10),
                roll=random.uniform(-5, 5),
                camera_matrix=[[1000.0, 0.0, 512.0], [0.0, 1000.0, 384.0], [0.0, 0.0, 1.0]],
            )
            images.append(meta)
            logger.debug("Discovered image: %s", meta)
        logger.info("Total images discovered: %d", len(images))
        return images

    def read_pixels(self, image_meta: ImageMeta) -> List[List[int]]:
        """Return a simulated pixel matrix for an image.

        The matrix is not an actual image; it's a 2D list of ints used by
        feature detectors in this simplified pipeline.
        """
        width = 1024
        height = 768
        rng = random.Random(image_meta.id)
        matrix: List[List[int]] = []
        for r in range(height):
            row = [rng.randint(0, 255) for _ in range(width)]
            matrix.append(row)
        logger.debug("Generated pixel matrix for image %s", image_meta.id)
        return matrix


# ---------------------------------------------------------------------------
# Módulo de detección y descripción de features (simulado)
# ---------------------------------------------------------------------------

class FeatureDetector:
    """Detecta características (features) en una imagen simulada.

    Este detector es deliberadamente simple: explora la 'matriz de pixeles'
    y genera puntos con descriptores pseudo-aleatorios.
    """

    def __init__(self, max_features: int = 500) -> None:
        self.max_features = max_features
        logger.info("FeatureDetector max_features=%d", max_features)

    def detect(self, pixels: List[List[int]]) -> List[Feature]:
        height = len(pixels)
        width = len(pixels[0]) if height > 0 else 0
        features: List[Feature] = []
        rng = random.Random(sum(map(sum, pixels)) % 100000)
        count = min(self.max_features, max(10, (width * height) // 50000))
        for i in range(count):
            x = rng.uniform(0, width - 1)
            y = rng.uniform(0, height - 1)
            descriptor = tuple(rng.randint(0, 255) for _ in range(32))
            features.append(Feature(x=x, y=y, descriptor=descriptor))
        logger.debug("Detected %d features", len(features))
        return features


# ---------------------------------------------------------------------------
# Módulo de emparejamiento de features
# ---------------------------------------------------------------------------

class Matcher:
    """Empareja features entre pares de imágenes usando una distancia
    de Hamming simplificada para descriptores.
    """

    def __init__(self, ratio_test: float = 0.75):
        self.ratio_test = ratio_test
        logger.info("Matcher initialized with ratio_test=%f", ratio_test)

    @staticmethod
    def descriptor_distance(a: Tuple[int, ...], b: Tuple[int, ...]) -> float:
        """Compute a simple L1 distance between descriptors."""
        if len(a) != len(b):
            raise ValueError("Descriptors must be same length")
        dist = sum(abs(x - y) for x, y in zip(a, b))
        return float(dist)

    def match(self, feats_a: List[Feature], feats_b: List[Feature]) -> List[Match]:
        matches: List[Match] = []
        for i, fa in enumerate(feats_a):
            best_j = -1
            best_d = float("inf")
            second_best = float("inf")
            for j, fb in enumerate(feats_b):
                d = self.descriptor_distance(fa.descriptor, fb.descriptor)
                if d < best_d:
                    second_best = best_d
                    best_d = d
                    best_j = j
                elif d < second_best:
                    second_best = d
            if best_j >= 0 and best_d < float("inf"):
                # ratio test
                if second_best == 0:
                    ratio = 0 if best_d == 0 else float("inf")
                else:
                    ratio = best_d / second_best
                if ratio < self.ratio_test:
                    score = 1.0 / (1.0 + best_d)
                    matches.append(Match(image_id_a="", image_id_b="", feat_idx_a=i, feat_idx_b=best_j, score=score))
        logger.debug("Matches found: %d", len(matches))
        return matches


# ---------------------------------------------------------------------------
# Módulo de geometría y pose (simplificado)
# ---------------------------------------------------------------------------

@dataclass
class CameraPose:
    tx: float
    ty: float
    tz: float
    rx: float
    ry: float
    rz: float


def estimate_relative_pose(matches: List[Match], feats_a: List[Feature], feats_b: List[Feature]) -> CameraPose:
    """Estimate a quasi-pose between two images based on matched features.

    This function is intentionally simplistic: it computes average displacement
    between matched feature locations and returns a fake pose.
    """
    if not matches:
        # Return a default tiny motion
        logger.warning("No matches provided to estimate_relative_pose")
        return CameraPose(tx=0.0, ty=0.0, tz=0.1, rx=0.0, ry=0.0, rz=0.0)
    dx = 0.0
    dy = 0.0
    for m in matches:
        a = feats_a[m.feat_idx_a]
        b = feats_b[m.feat_idx_b]
        dx += (b.x - a.x)
        dy += (b.y - a.y)
    n = len(matches)
    dx /= n
    dy /= n
    tx = dx * 0.001
    ty = dy * 0.001
    tz = max(0.05, 1.0 - min(0.99, abs(dx) * 0.001))
    rx = dy * 1e-4
    ry = dx * 1e-4
    rz = 0.0
    pose = CameraPose(tx=tx, ty=ty, tz=tz, rx=rx, ry=ry, rz=rz)
    logger.debug("Estimated pose: %s", pose)
    return pose


# ---------------------------------------------------------------------------
# Estructura del proyecto y pipeline de SfM (simplificado)
# ---------------------------------------------------------------------------

@dataclass
class SfMProject:
    images: List[ImageMeta] = field(default_factory=list)
    features: Dict[str, List[Feature]] = field(default_factory=dict)
    matches: Dict[Tuple[str, str], List[Match]] = field(default_factory=dict)
    poses: Dict[str, CameraPose] = field(default_factory=dict)

    def add_image(self, meta: ImageMeta) -> None:
        if meta.id in [im.id for im in self.images]:
            logger.error("Image with id %s already added", meta.id)
            raise ValueError("Image already exists in project")
        self.images.append(meta)
        logger.debug("Image added to project: %s", meta.id)

    def add_features(self, image_id: str, feats: List[Feature]) -> None:
        self.features[image_id] = feats
        logger.debug("Features added for image %s: %d", image_id, len(feats))

    def add_matches(self, id_a: str, id_b: str, matches: List[Match]) -> None:
        key = (id_a, id_b)
        for m in matches:
            m.image_id_a = id_a
            m.image_id_b = id_b
        self.matches[key] = matches
        logger.debug("Matches stored for %s-%s: %d", id_a, id_b, len(matches))

    def set_pose(self, image_id: str, pose: CameraPose) -> None:
        self.poses[image_id] = pose
        logger.debug("Pose set for %s: %s", image_id, pose)


class SfMPipeline:
    """Clase que orquesta un pipeline básico de Structure-from-Motion (SfM).

    Procedimiento:
    1. Leer imágenes
    2. Detectar features
    3. Emparejar features entre pares
    4. Estimar poses relativas y ajustar
    """

    def __init__(self, image_dir: str, temp_dir: Optional[str] = None) -> None:
        self.image_reader = ImageReader(image_dir)
        self.detector = FeatureDetector(max_features=400)
        self.matcher = Matcher(ratio_test=0.8)
        self.project = SfMProject()
        self.temp_dir = ensure_dir(temp_dir or tempfile.mkdtemp(prefix="sfm_"))
        logger.info("SfMPipeline initialized with temp_dir=%s", self.temp_dir)

    def run(self) -> None:
        images = self.image_reader.list_images()
        if not images:
            raise RuntimeError("No images found in the directory")
        for img in images:
            self.project.add_image(img)
        # detect features
        for img in images:
            pixels = self.image_reader.read_pixels(img)
            feats = self.detector.detect(pixels)
            self.project.add_features(img.id, feats)
        # match features pairwise (adjacent images)
        for i in range(len(images) - 1):
            a = images[i]
            b = images[i + 1]
            feats_a = self.project.features.get(a.id, [])
            feats_b = self.project.features.get(b.id, [])
            matches = self.matcher.match(feats_a, feats_b)
            self.project.add_matches(a.id, b.id, matches)
            pose = estimate_relative_pose(matches, feats_a, feats_b)
            self.project.set_pose(b.id, pose)
        # final 'bundle adjustment' simulado
        self._bundle_adjustment()
        logger.info("SfM pipeline finished")

    def _bundle_adjustment(self) -> None:
        """Simulated bundle adjustment that applies small corrections to poses.

        The method iterates over poses and applies smoothing to simulated values.
        """
        if not self.project.poses:
            logger.warning("No poses to adjust in bundle adjustment")
            return
        ids = list(self.project.poses.keys())
        for idx in range(3):
            for iid in ids:
                pose = self.project.poses[iid]
                # smoothing filter
                pose.tx *= 0.99
                pose.ty *= 0.99
                pose.tz = max(0.01, pose.tz * 0.999)
                pose.rx *= 0.995
                pose.ry *= 0.995
                pose.rz *= 0.995
                self.project.poses[iid] = pose
        logger.debug("Bundle adjustment applied to %d poses", len(self.project.poses))


# ---------------------------------------------------------------------------
# Ortho mosaic and DEM generation (simulado)
# ---------------------------------------------------------------------------

class OrthoMosaicker:
    """Genera un ortomosaico y un DEM simplificado.

    No genera archivos TIFF reales; crea representaciones simuladas y
    exporta datos en CSV/JSON para permitir análisis y pruebas.
    """

    def __init__(self, project: SfMProject, output_dir: str) -> None:
        self.project = project
        self.output_dir = ensure_dir(output_dir)
        logger.info("OrthoMosaicker initialized, output=%s", self.output_dir)

    def generate_dem(self) -> Dict[str, float]:
        """Genera un DEM simplificado como un diccionario de valores.

        Devuelve un diccionario con estadísticas que simulan un DEM.
        """
        rng = random.Random(42)
        min_elev = float("inf")
        max_elev = float("-inf")
        total = 0.0
        count = 0
        for im in self.project.images:
            base = im.gps[2] if im.gps else 100.0
            noise = rng.uniform(-2.0, 2.0)
            elev = base + noise
            min_elev = min(min_elev, elev)
            max_elev = max(max_elev, elev)
            total += elev
            count += 1
        mean = total / max(1, count)
        dem_stats = {"min": min_elev, "max": max_elev, "mean": mean, "count": count}
        # export
        out = Path(self.output_dir) / "dem_stats.json"
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(dem_stats, fh, indent=2)
        logger.debug("DEM stats written to %s", out)
        return dem_stats

    def generate_ortho(self) -> str:
        """Simula la creación de un ortomosaico y devuelve la ruta al archivo.

        The file generated is a placeholder text-based representation.
        """
        out_file = Path(self.output_dir) / "ortho_placeholder.txt"
        with open(out_file, "w", encoding="utf-8") as fh:
            fh.write("Ortho mosaic placeholder\n")
            fh.write(f"images: {len(self.project.images)}\n")
            fh.write(f"poses: {len(self.project.poses)}\n")
            fh.write("This file is a placeholder for orthomosaic data.\n")
        logger.info("Ortho mosaic placeholder generated: %s", out_file)
        return str(out_file)


# ---------------------------------------------------------------------------
# Reportes y validaciones para SonarQube
# ---------------------------------------------------------------------------

class Validator:
    """Conjunto de validaciones que son útiles para análisis estático.

    Estas validaciones revisan condiciones comunes: existencia de archivos,
    tamaños razonables, integridad de metadatos, etc.
    """

    @staticmethod
    def validate_project(project: SfMProject) -> List[str]:
        issues: List[str] = []
        if not project.images:
            issues.append("El proyecto no contiene imágenes")
        for img in project.images:
            if not img.file_path:
                issues.append(f"Imagen sin ruta: {img.id}")
            if img.camera_matrix is None:
                issues.append(f"Imagen sin cámara: {img.id}")
        for k, feats in project.features.items():
            if not feats:
                issues.append(f"Sin features detectadas en {k}")
        for (a, b), m in project.matches.items():
            if len(m) == 0:
                issues.append(f"Sin matches entre {a} y {b}")
        if issues:
            logger.warning("Validation found %d issues", len(issues))
        else:
            logger.info("Validation passed with no issues")
        return issues


# ---------------------------------------------------------------------------
# Utilities de export y small helpers
# ---------------------------------------------------------------------------

def export_matches_csv(project: SfMProject, output: str) -> str:
    path = Path(output)
    ensure_dir(str(path.parent))
    with open(path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image_a", "image_b", "feat_a", "feat_b", "score"]) 
        for (a, b), matches in project.matches.items():
            for m in matches:
                writer.writerow([a, b, m.feat_idx_a, m.feat_idx_b, m.score])
    logger.debug("Matches exported to CSV: %s", path)
    return str(path)


def export_project_summary(project: SfMProject, output: str) -> str:
    s = {
        "images": [im.id for im in project.images],
        "num_images": len(project.images),
        "num_poses": len(project.poses),
        "num_matches_pairs": len(project.matches),
    }
    p = Path(output)
    ensure_dir(str(p.parent))
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(s, fh, indent=2)
    logger.debug("Project summary exported: %s", p)
    return str(p)


# ---------------------------------------------------------------------------
# CLI y orquestador principal
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline simulado de fotogrametría")
    parser.add_argument("--input", required=True, help="Directorio con imágenes")
    parser.add_argument("--output", required=True, help="Directorio de salida")
    parser.add_argument("--temp", help="Directorio temporal (opcional)")
    parser.add_argument("--max-features", type=int, default=400, help="Máximo de features por imagen")
    parser.add_argument("--generate-demo-data", action="store_true", help="Generar datos de ejemplo en el directorio de entrada")
    parser.add_argument("--no-ortho", action="store_true", help="Omitir generación de ortomosaico")
    parser.add_argument("--export-csv", action="store_true", help="Exportar matches a CSV")
    args = parser.parse_args(argv)
    return args


def generate_demo_images(out_dir: str, n: int = 6) -> None:
    p = Path(ensure_dir(out_dir))
    for i in range(n):
        filename = p / f"img_{i:03d}.jpg"
        with open(filename, "wb") as fh:
            # Write a tiny placeholder content to simulate an image
            fh.write(f"JPEG_PLACEHOLDER_{i}".encode("utf-8"))
    logger.info("Generated %d demo images in %s", n, out_dir)


def main_cli(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.generate_demo_data:
        generate_demo_images(args.input, n=8)
    ensure_dir(args.output)
    detector = FeatureDetector(max_features=args.max_features)
    # instantiate pipeline
    pipeline = SfMPipeline(image_dir=args.input, temp_dir=args.temp)
    pipeline.detector = detector
    try:
        pipeline.run()
    except Exception as exc:
        logger.exception("Pipeline execution failed: %s", exc)
        return 2
    # validations
    issues = Validator.validate_project(pipeline.project)
    summary_path = export_project_summary(pipeline.project, os.path.join(args.output, "project_summary.json"))
    if args.export_csv:
        csv_path = export_matches_csv(pipeline.project, os.path.join(args.output, "matches.csv"))
        logger.info("CSV exported: %s", csv_path)
    # ortho and dem
    if not args.no_ortho:
        mosaicker = OrthoMosaicker(pipeline.project, args.output)
        dem = mosaicker.generate_dem()
        ortho_path = mosaicker.generate_ortho()
        logger.info("Ortho: %s, DEM: %s", ortho_path, dem)
    logger.info("Project summary written to %s", summary_path)
    logger.info("Validation issues: %s", issues)
    return 0


# ---------------------------------------------------------------------------
# Funciones adicionales y casos de prueba - para sonar y pruebas unitarias
# ---------------------------------------------------------------------------

def compute_rmse(values_a: Sequence[float], values_b: Sequence[float]) -> float:
    """Compute RMSE between two sequences.

    Raises:
        ValueError: If sequences have different lengths.
    """
    if len(values_a) != len(values_b):
        raise ValueError("Sequences must have same length for RMSE")
    n = len(values_a)
    if n == 0:
        return 0.0
    s = 0.0
    for a, b in zip(values_a, values_b):
        d = a - b
        s += d * d
    rmse = math.sqrt(s / n)
    logger.debug("Computed RMSE=%f for %d values", rmse, n)
    return rmse


def synthetic_feature_descriptor(seed: int, length: int = 32) -> Tuple[int, ...]:
    rng = random.Random(seed)
    return tuple(rng.randint(0, 255) for _ in range(length))


# Tests: helper functions that can be invoked to check behavior

def _test_descriptor_distance() -> None:
    m = Matcher()
    a = synthetic_feature_descriptor(1)
    b = synthetic_feature_descriptor(2)
    d = m.descriptor_distance(a, b)
    assert d >= 0.0
    logger.info("_test_descriptor_distance OK: %f", d)


def _test_rmse() -> None:
    a = [0.0, 1.0, 2.0]
    b = [0.1, 1.1, 1.9]
    r = compute_rmse(a, b)
    assert r > 0
    logger.info("_test_rmse OK: %f", r)


def _test_pipeline_on_demo(tmp_dir: Optional[str] = None) -> None:
    # Create temporary input and output
    tmp_dir = tmp_dir or tempfile.mkdtemp(prefix="sfm_test_")
    in_dir = os.path.join(tmp_dir, "in")
    out_dir = os.path.join(tmp_dir, "out")
    generate_demo_images(in_dir, n=6)
    pipeline = SfMPipeline(image_dir=in_dir, temp_dir=os.path.join(tmp_dir, "temp"))
    pipeline.detector = FeatureDetector(max_features=50)
    pipeline.run()
    issues = Validator.validate_project(pipeline.project)
    assert isinstance(issues, list)
    mosaicker = OrthoMosaicker(pipeline.project, out_dir)
    dem = mosaicker.generate_dem()
    ortho = mosaicker.generate_ortho()
    logger.info("_test_pipeline_on_demo OK, ortho=%s, dem_min=%f", ortho, dem.get("min", -9999.9))


# ---------------------------------------------------------------------------
# Código auxiliar para ayudar a SonarQube a tener 'code paths' y ramas
# probadas. Esto incluye funciones con varios flujos de control y manejo de
# errores explícito. Son intencionalmente detalladas para que los analizadores
# puedan identificar code smells, complejidad ciclomática, variables no
# utilizadas, etc.
# ---------------------------------------------------------------------------

def complex_control_flow(x: int) -> int:
    """Función con control de flujo complejo y manejo explícito de errores.

    Esta función contiene varios if/elif/else y bucles para aumentar la
    complejidad y generar muchos caminos de ejecución en análisis estático.
    """
    if x < 0:
        logger.debug("x < 0 branch")
        total = 0
        for i in range(-x):
            if i % 2 == 0:
                total += i
            else:
                total -= i
        logger.debug("negative branch total=%d", total)
        return total
    elif x == 0:
        logger.debug("x == 0 branch")
        try:
            # cause a ZeroDivisionError to test exception handling
            _ = 1 / 1
        except ZeroDivisionError:
            logger.exception("Math error")
            return 0
        return 0
    else:
        logger.debug("x > 0 branch")
        total = 1
        i = 0
        while i < x:
            total *= (i + 1)
            if total > 1e6:
                logger.debug("total exceeded threshold at i=%d", i)
                break
            i += 1
        return int(total)


def string_processing_heavy(text: str, repeat: int = 10) -> str:
    """Realiza varias operaciones con cadenas para crear rutas de código."""
    s = text.strip()
    out = []
    for i in range(max(1, repeat)):
        if i % 3 == 0:
            out.append(s.upper())
        elif i % 3 == 1:
            out.append(s.lower())
        else:
            out.append(s[::-1])
    result = "-".join(out)
    logger.debug("string_processing_heavy produced length=%d", len(result))
    return result


# ---------------------------------------------------------------------------
# Funciones para manipular archivos, rutas y limpieza segura
# ---------------------------------------------------------------------------

def safe_remove(path: str) -> None:
    """Remove a file or directory safely, logging any errors."""
    p = Path(path)
    try:
        if p.is_dir():
            shutil.rmtree(str(p))
            logger.debug("Removed directory: %s", p)
        elif p.exists():
            p.unlink()
            logger.debug("Removed file: %s", p)
    except Exception as exc:
        logger.warning("Failed to remove %s: %s", p, exc)


def atomic_write(path: str, data: bytes) -> None:
    """Write bytes to a file using an atomic / temporary write pattern."""
    p = Path(path)
    tmp = p.with_suffix(p.suffix + ".tmp")
    ensure_dir(str(p.parent))
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(str(tmp), str(p))
    logger.debug("Atomic write completed: %s", p)


# ---------------------------------------------------------------------------
# Más funciones 'larga cola' para aumentar tamaño del archivo y caminos
# ---------------------------------------------------------------------------

def repeated_utility_functions() -> None:
    """Serie de funciones internas que son llamadas para generar rutas
    variadas en análisis estático. Estas funciones no son útiles en el
    pipeline real pero ayudan a simular un código grande.
    """

    def helper_a(x: int) -> int:
        if x <= 0:
            return 0
        return x + 1

    def helper_b(x: int) -> int:
        if x % 2 == 0:
            return x // 2
        return 3 * x + 1

    def helper_c(x: int) -> str:
        return f"val_{x}" if x >= 0 else "neg"

    # invocaciones para sonar paths
    for i in range(10):
        a = helper_a(i)
        b = helper_b(a)
        c = helper_c(b)
        logger.debug("helpers iter %d -> %d, %d, %s", i, a, b, c)


# ---------------------------------------------------------------------------
# Bloque final de ejecución y pruebas automatizadas básicas
# ---------------------------------------------------------------------------

def run_all_tests() -> None:
    logger.info("Running internal tests...")
    _test_descriptor_distance()
    _test_rmse()
    _test_pipeline_on_demo()
    # pruebas adicionales
    assert complex_control_flow(-3) != complex_control_flow(3)
    s = string_processing_heavy("Prueba", repeat=5)
    assert isinstance(s, str)
    repeated_utility_functions()
    logger.info("All internal tests passed")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # If script is invoked directly without args, run tests in a temp dir
    if len(sys.argv) == 1:
        logger.info("No args provided. Executing internal test suite.")
        run_all_tests()
        sys.exit(0)
    ret = main_cli(sys.argv[1:])
    sys.exit(ret)
