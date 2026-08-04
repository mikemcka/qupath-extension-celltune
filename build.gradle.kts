plugins {
    // To create a shadow/fat jar that bundles non-core dependencies
    id("com.gradleup.shadow") version "8.3.5"
    // QuPath Gradle extension convention plugin
    id("qupath-conventions")
    // Static analysis — reporting only, never fails the build (see config below)
    id("com.github.spotbugs") version "6.4.2"
    // Code formatting — palantir-java-format, non-blocking (see config below)
    id("com.diffplug.spotless") version "7.0.2"
}

qupathExtension {
    name = "qupath-extension-celltune"
    group = "io.github.qupath"
    version = "0.2.4"
    description = "CellTune-style active learning cell classifier for QuPath"
    automaticModule = "io.github.qupath.extension.celltune"
}

dependencies {

    // Main dependencies for most QuPath extensions
    shadow(libs.bundles.qupath)
    shadow(libs.bundles.logging)
    shadow(libs.qupath.fxtras)

    // ML models — bundled into the shadow jar
    implementation("ml.dmlc:xgboost4j_2.13:2.1.4")
    shadow("ml.dmlc:xgboost4j_2.13:2.1.4")
    implementation("io.github.metarank:lightgbm4j:4.6.0-2")
    shadow("io.github.metarank:lightgbm4j:4.6.0-2")

    // SMILE — PCA / UMAP / k-means for the cell scatter plot. Bundled into the
    // shadow jar. NOTE: SMILE's PCA (Matrix.svd) and UMAP (spectral init) call
    // into native BLAS/LAPACK via org.bytedeco openblas + arpack-ng — there is
    // no pure-Java fallback in 3.1.1, so the bytedeco natives must be bundled.
    // This adds the platform native binaries (multiple OS classifiers) to the
    // jar; accepted to keep PCA/UMAP working and cross-platform portable.
    implementation("com.github.haifengl:smile-core:3.1.1")
    shadow("com.github.haifengl:smile-core:3.1.1")

    // CWTS networkanalysis — the reference Leiden / Louvain community-detection
    // implementation (Traag, van Eck & Waltman; same authors as the Python
    // leidenalg that scanpy/scimap use). Smile has no graph clustering. Used for
    // graph-based phenotype clustering in the scatter-plot workflow. Bundled.
    implementation("nl.cwts:networkanalysis:1.3.0")
    shadow("nl.cwts:networkanalysis:1.3.0")

    // HNSW — approximate-NN kNN graph construction for Leiden at cohort scale
    // (tens of millions of cells). Pure Java, no native binaries (unlike Smile's
    // OpenBLAS/ARPACK bundling above) — see .planning/phases/15-all-cells-leiden-clustering/15-RESEARCH.md "Standard Stack".
    implementation("com.github.jelmerk:hnswlib-core:1.2.1")
    shadow("com.github.jelmerk:hnswlib-core:1.2.1")

    // For testing
    testImplementation(libs.bundles.qupath)
    testImplementation(libs.junit)
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}

tasks.jar {
    manifest {
        attributes(
            "Implementation-Version" to project.version
        )
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────
// Forward `-Dcelltune.*` from the Gradle CLI into the test JVM. Gradle does not
// propagate system properties to forked test workers, and the performance
// benchmarks (e.g. ResamplerBenchmark) are gated on `celltune.bench` so they
// stay out of normal `./gradlew test` runs.
tasks.test {
    System.getProperties().forEach { key, value ->
        val name = key.toString()
        if (name.startsWith("celltune.")) systemProperty(name, value.toString())
    }
}

// ── Static analysis (SpotBugs) ──────────────────────────────────────────────
// Reporting only: establishes a baseline of potential bugs without blocking the
// build. Run `./gradlew spotbugsMain` and open the HTML report (path printed
// below) under build/reports/spotbugs/. Tune over time; never wire into `check`
// as a failing gate until the baseline is triaged.
spotbugs {
    ignoreFailures.set(true)        // never fail the build on findings
    showStackTraces.set(false)
    // MEDIUM+ confidence keeps the first baseline signal-to-noise reasonable.
    effort.set(com.github.spotbugs.snom.Effort.DEFAULT)
    reportLevel.set(com.github.spotbugs.snom.Confidence.MEDIUM)
}

// Analyse main sources only; test sources are not the audit target.
tasks.spotbugsMain {
    reports.create("html") { required.set(true) }
}
tasks.named("spotbugsTest") {
    enabled = false
}

// ── Code formatting (Spotless + palantir-java-format) ───────────────────────
// Auto-formatter. Run `./gradlew spotlessApply` to reformat in place, or
// `./gradlew spotlessCheck` to verify without editing. palantir-java-format =
// 4-space indent, 120-col lines (closest to the existing style, smallest diff).
//
// Enforced: `isEnforceCheck = true` wires spotlessCheck into `check`, so an
// unformatted tree fails `./gradlew check` (and CI). Run `spotlessApply` before
// committing. Set this back to `false` to make formatting reporting-only.
spotless {
    isEnforceCheck = true
    java {
        target("src/*/java/**/*.java")
        palantirJavaFormat("2.93.0")
        removeUnusedImports()
    }
}
