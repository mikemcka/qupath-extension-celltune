package qupath.ext.spclassify.model;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Whole-image, per-channel pixel statistics computed on a low-resolution version
 * of a (typically pyramidal) image.
 * <p>
 * <b>Attribution:</b> the set of whole-image summary statistics computed here, and the idea of
 * using them as a cheap image-quality prescreen, were adapted from
 * <a href="https://pypi.org/project/pixel-patrol/">pixel-patrol</a> (MIT licence). This Java
 * implementation is original to this project.
 * <p>
 * This class is the pixel-level twin of the cell-composition analytics in
 * {@link CohortAnomalyAnalyzer}: instead of summarising an image by the classes
 * of its cells, it summarises an image by the intensity distribution of its raw
 * pixels. It is intended as a cheap <em>prescreen</em> run at the start of a
 * project to surface images that are mostly background, over-exposed, weakly
 * stained, or otherwise unusual — before any segmentation or classification
 * exists.
 * <p>
 * The numeric computation is intentionally free of any QuPath or JavaFX
 * dependency so it can be unit-tested in isolation. Callers ({@code
 * ImagePixelStatsReader}) read a downsampled region from an
 * {@code ImageServer}, unpack it into one {@code float[]} per channel, and hand
 * the arrays here.
 *
 * <h2>Statistic definitions</h2>
 * For one channel, over all {@code N} pixels of the low-resolution image
 * (values sorted ascending where percentiles are involved):
 * <ul>
 *   <li><b>mean</b> — arithmetic mean {@code Σx / N}. Sensitive to outliers by
 *       design; kept for parity with cell-level marker means.</li>
 *   <li><b>std</b> — population standard deviation
 *       {@code sqrt(Σ(x − mean)² / N)}.</li>
 *   <li><b>min / max</b> — extrema. {@code max} is reported but should not be
 *       used as a sort key: a single hot pixel moves it.</li>
 *   <li><b>median</b> — 50th percentile (linear interpolation). The robust
 *       brightness measure and the recommended sort key.</li>
 *   <li><b>p1 / p99</b> — 1st / 99th percentiles. {@code p1} is the noise floor,
 *       {@code p99} the true signal ceiling; both ignore single extreme
 *       pixels.</li>
 *   <li><b>saturationFraction</b> — fraction of pixels at or above
 *       {@code 0.999 × dtypeMax} (clipping / over-exposure). {@code NaN} when the
 *       pixel type has no fixed maximum (floating-point images).</li>
 *   <li><b>otsuThreshold</b> — the foreground/background split computed from the
 *       channel histogram by Otsu's method.</li>
 *   <li><b>backgroundFraction</b> — fraction of pixels strictly below
 *       {@code otsuThreshold}.</li>
 *   <li><b>foregroundCoverage</b> — {@code 1 − backgroundFraction}; the direct
 *       "how much real signal" measure.</li>
 *   <li><b>dynamicRange</b> — {@code p99 − p1}; flat / weak / empty channels
 *       score near zero.</li>
 *   <li><b>laplacianVariance</b> — variance of the discrete Laplacian (a focus /
 *       sharpness proxy): high for crisp, in-focus structure and low for blurred
 *       or out-of-focus images. It is intensity-scale dependent, so it is best
 *       compared within a cohort acquired the same way. {@code NaN} when the
 *       spatial shape is unknown or the image is smaller than 3×3.</li>
 * </ul>
 * The image-level <b>emptyFraction</b> is the fraction of pixels that are below
 * their channel's Otsu threshold in <em>every</em> channel — the single best
 * "this slide is mostly glass/background" indicator.
 */
public final class ImagePixelStats {

    /** Number of histogram bins used for Otsu thresholding. */
    private static final int OTSU_BINS = 256;

    /** Fraction of {@code dtypeMax} at or above which a pixel counts as saturated. */
    private static final double SATURATION_FRACTION_OF_MAX = 0.999;

    private ImagePixelStats() {}

    /**
     * Immutable per-channel statistics. See {@link ImagePixelStats} for the
     * definition of each field.
     *
     * @param channel            channel display name
     * @param pixelCount         number of pixels contributing (finite values only)
     * @param mean               arithmetic mean
     * @param std                population standard deviation
     * @param min                minimum value
     * @param median             50th percentile
     * @param max                maximum value
     * @param p1                 1st percentile (noise floor)
     * @param p99                99th percentile (signal ceiling)
     * @param saturationFraction fraction of clipped pixels ({@code NaN} for float types)
     * @param otsuThreshold      Otsu foreground/background split
     * @param backgroundFraction fraction below {@code otsuThreshold}
     * @param foregroundCoverage {@code 1 − backgroundFraction}
     * @param dynamicRange       {@code p99 − p1}
     * @param laplacianVariance  variance of the discrete Laplacian (focus proxy);
     *                           {@code NaN} when the spatial shape is unknown
     */
    public record ChannelStats(
            String channel,
            long pixelCount,
            double mean,
            double std,
            double min,
            double median,
            double max,
            double p1,
            double p99,
            double saturationFraction,
            double otsuThreshold,
            double backgroundFraction,
            double foregroundCoverage,
            double dynamicRange,
            double laplacianVariance) {}

    /**
     * Immutable whole-image summary: one {@link ChannelStats} per channel plus
     * the cross-channel {@link #emptyFraction()}.
     *
     * @param imageName     image name
     * @param downsample    downsample factor actually used to read the pixels
     * @param width         width in pixels of the low-resolution image read
     * @param height        height in pixels of the low-resolution image read
     * @param channels      per-channel statistics, in channel order
     * @param emptyFraction fraction of pixels below the Otsu threshold in every channel
     */
    public record ImageStats(
            String imageName,
            double downsample,
            int width,
            int height,
            List<ChannelStats> channels,
            double emptyFraction) {

        public ImageStats {
            imageName = imageName == null ? "" : imageName;
            channels = channels == null ? List.of() : List.copyOf(channels);
        }
    }

    /**
     * Compute statistics for a single channel.
     *
     * @param name     channel display name
     * @param values   pixel values for this channel (may contain {@code NaN},
     *                  which are ignored); not modified
     * @param dtypeMax maximum representable value for the pixel type (e.g. 255,
     *                  65535), or {@code NaN} / non-positive for floating-point
     *                  images where saturation is undefined
     * @return the channel statistics; a zeroed record if {@code values} is empty
     */
    public static ChannelStats computeChannel(String name, float[] values, double dtypeMax) {
        return computeChannel(name, values, dtypeMax, 0, 0);
    }

    /**
     * Compute statistics for a single channel, additionally computing the
     * focus / sharpness proxy ({@link ChannelStats#laplacianVariance()}) from the
     * 2D pixel layout.
     *
     * @param name     channel display name
     * @param values   row-major pixel values for this channel ({@code NaN}
     *                  ignored); not modified
     * @param dtypeMax maximum representable value for the pixel type, or
     *                  {@code NaN} / non-positive for floating-point images
     * @param width    image width in pixels; pass {@code 0} (with {@code height}
     *                  0) to skip the focus metric, leaving it {@code NaN}
     * @param height   image height in pixels
     * @return the channel statistics; a zeroed record if {@code values} is empty
     */
    public static ChannelStats computeChannel(String name, float[] values, double dtypeMax, int width, int height) {
        String channel = name == null ? "" : name;
        if (values == null || values.length == 0) {
            return new ChannelStats(
                    channel,
                    0L,
                    Double.NaN,
                    Double.NaN,
                    Double.NaN,
                    Double.NaN,
                    Double.NaN,
                    Double.NaN,
                    Double.NaN,
                    Double.NaN,
                    Double.NaN,
                    Double.NaN,
                    Double.NaN,
                    Double.NaN,
                    Double.NaN);
        }

        // Copy finite values into a double[] we can sort for percentiles.
        double[] sorted = new double[values.length];
        int n = 0;
        double sum = 0.0;
        for (float raw : values) {
            if (Float.isNaN(raw) || Float.isInfinite(raw)) {
                continue;
            }
            double v = raw;
            sorted[n++] = v;
            sum += v;
        }
        if (n == 0) {
            return new ChannelStats(
                    channel,
                    0L,
                    Double.NaN,
                    Double.NaN,
                    Double.NaN,
                    Double.NaN,
                    Double.NaN,
                    Double.NaN,
                    Double.NaN,
                    Double.NaN,
                    Double.NaN,
                    Double.NaN,
                    Double.NaN,
                    Double.NaN,
                    Double.NaN);
        }
        if (n < sorted.length) {
            sorted = Arrays.copyOf(sorted, n);
        }
        Arrays.sort(sorted);

        double mean = sum / n;
        double sqSum = 0.0;
        for (double v : sorted) {
            double d = v - mean;
            sqSum += d * d;
        }
        double std = Math.sqrt(sqSum / n);

        double min = sorted[0];
        double max = sorted[n - 1];
        double median = percentileSorted(sorted, 50.0);
        double p1 = percentileSorted(sorted, 1.0);
        double p99 = percentileSorted(sorted, 99.0);

        double saturationFraction = Double.NaN;
        if (dtypeMax > 0.0 && !Double.isNaN(dtypeMax)) {
            double satCutoff = SATURATION_FRACTION_OF_MAX * dtypeMax;
            int saturated = 0;
            // sorted ascending — count from the top.
            for (int i = n - 1; i >= 0; i--) {
                if (sorted[i] >= satCutoff) {
                    saturated++;
                } else {
                    break;
                }
            }
            saturationFraction = (double) saturated / n;
        }

        double otsu = otsuThresholdSorted(sorted, min, max);
        long below = countBelow(sorted, otsu);
        double backgroundFraction = (double) below / n;
        double foregroundCoverage = 1.0 - backgroundFraction;
        double dynamicRange = p99 - p1;
        double laplacianVariance = laplacianVariance(values, width, height);

        return new ChannelStats(
                channel,
                n,
                mean,
                std,
                min,
                median,
                max,
                p1,
                p99,
                saturationFraction,
                otsu,
                backgroundFraction,
                foregroundCoverage,
                dynamicRange,
                laplacianVariance);
    }

    /**
     * Variance of the discrete 4-neighbour Laplacian over the interior pixels of
     * a row-major image — a standard no-reference focus / sharpness measure
     * (higher = sharper). Pixels whose 3×3 cross neighbourhood contains a
     * non-finite value are skipped.
     *
     * @param values row-major pixel values, length ≥ {@code width × height}
     * @param width  image width in pixels
     * @param height image height in pixels
     * @return the Laplacian variance, or {@code NaN} when the shape is unknown,
     *         smaller than 3×3, or no interior pixel has a finite neighbourhood
     */
    public static double laplacianVariance(float[] values, int width, int height) {
        if (values == null || width < 3 || height < 3 || (long) width * height > values.length) {
            return Double.NaN;
        }
        double sum = 0.0;
        double sumSq = 0.0;
        long count = 0;
        for (int y = 1; y < height - 1; y++) {
            int row = y * width;
            for (int x = 1; x < width - 1; x++) {
                int i = row + x;
                float c = values[i];
                float up = values[i - width];
                float down = values[i + width];
                float left = values[i - 1];
                float right = values[i + 1];
                if (!Float.isFinite(c)
                        || !Float.isFinite(up)
                        || !Float.isFinite(down)
                        || !Float.isFinite(left)
                        || !Float.isFinite(right)) {
                    continue;
                }
                double lap = (double) up + down + left + right - 4.0 * c;
                sum += lap;
                sumSq += lap * lap;
                count++;
            }
        }
        if (count == 0) {
            return Double.NaN;
        }
        double mean = sum / count;
        double var = sumSq / count - mean * mean;
        return Math.max(0.0, var);
    }

    /**
     * Compute the full whole-image summary from per-channel pixel arrays.
     *
     * @param imageName     image name
     * @param downsample    downsample factor used to read the pixels
     * @param width         low-resolution image width
     * @param height        low-resolution image height
     * @param channelNames  channel display names; index {@code c} pairs with
     *                      {@code channelValues[c]}
     * @param channelValues per-channel pixel values, all of equal length
     *                      (the pixel count of the low-resolution image)
     * @param dtypeMax      maximum representable pixel value, or {@code NaN} for
     *                      floating-point images
     * @return the image statistics
     */
    public static ImageStats compute(
            String imageName,
            double downsample,
            int width,
            int height,
            List<String> channelNames,
            float[][] channelValues,
            double dtypeMax) {

        if (channelValues == null || channelValues.length == 0) {
            return new ImageStats(imageName, downsample, width, height, List.of(), Double.NaN);
        }

        int nChannels = channelValues.length;
        var channelStats = new ArrayList<ChannelStats>(nChannels);
        double[] otsuThresholds = new double[nChannels];
        for (int c = 0; c < nChannels; c++) {
            String name =
                    (channelNames != null && c < channelNames.size()) ? channelNames.get(c) : ("Channel " + (c + 1));
            ChannelStats cs = computeChannel(name, channelValues[c], dtypeMax, width, height);
            channelStats.add(cs);
            otsuThresholds[c] = cs.otsuThreshold();
        }

        double emptyFraction = emptyFraction(channelValues, otsuThresholds);

        return new ImageStats(imageName, downsample, width, height, channelStats, emptyFraction);
    }

    /**
     * Fraction of pixels that are below their channel's Otsu threshold in every
     * channel (i.e. background everywhere — "empty" / glass).
     *
     * @param channelValues  per-channel pixel arrays of equal length
     * @param otsuThresholds per-channel Otsu thresholds (same length as
     *                       {@code channelValues})
     * @return the empty fraction in {@code [0, 1]}, or {@code NaN} if there are
     *         no pixels or channels
     */
    public static double emptyFraction(float[][] channelValues, double[] otsuThresholds) {
        if (channelValues == null
                || channelValues.length == 0
                || otsuThresholds == null
                || otsuThresholds.length == 0) {
            return Double.NaN;
        }
        int nChannels = Math.min(channelValues.length, otsuThresholds.length);
        int nPixels = Integer.MAX_VALUE;
        for (int c = 0; c < nChannels; c++) {
            if (channelValues[c] == null) {
                return Double.NaN;
            }
            nPixels = Math.min(nPixels, channelValues[c].length);
        }
        if (nPixels <= 0) {
            return Double.NaN;
        }

        long empty = 0;
        for (int i = 0; i < nPixels; i++) {
            boolean backgroundEverywhere = true;
            for (int c = 0; c < nChannels; c++) {
                double t = otsuThresholds[c];
                float v = channelValues[c][i];
                // A NaN threshold means the channel is uninformative; treat it as
                // not contributing a foreground vote (i.e. it never rescues a pixel
                // from being "empty"). A pixel at/above threshold is foreground.
                if (!Double.isNaN(t) && !Float.isNaN(v) && v >= t) {
                    backgroundEverywhere = false;
                    break;
                }
            }
            if (backgroundEverywhere) {
                empty++;
            }
        }
        return (double) empty / nPixels;
    }

    /**
     * Linear-interpolated percentile of an already-sorted (ascending) array.
     *
     * @param sorted     ascending-sorted values, length ≥ 1
     * @param percentile percentile in {@code [0, 100]}
     * @return the interpolated percentile value
     */
    public static double percentileSorted(double[] sorted, double percentile) {
        int n = sorted.length;
        if (n == 0) {
            return Double.NaN;
        }
        if (n == 1) {
            return sorted[0];
        }
        double p = Math.max(0.0, Math.min(100.0, percentile));
        double rank = (p / 100.0) * (n - 1);
        int lo = (int) Math.floor(rank);
        int hi = (int) Math.ceil(rank);
        if (lo == hi) {
            return sorted[lo];
        }
        double frac = rank - lo;
        return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
    }

    /**
     * Otsu's threshold over an ascending-sorted array, computed from a
     * fixed-width histogram between {@code min} and {@code max}.
     *
     * @param sorted ascending-sorted values, length ≥ 1
     * @param min    minimum value (== {@code sorted[0]})
     * @param max    maximum value (== {@code sorted[length-1]})
     * @return the threshold intensity that maximises between-class variance, or
     *         {@code min} for a degenerate (single-value) distribution
     */
    public static double otsuThresholdSorted(double[] sorted, double min, double max) {
        int n = sorted.length;
        if (n == 0) {
            return Double.NaN;
        }
        double range = max - min;
        if (range <= 0.0) {
            // All pixels equal — no meaningful split; nothing is below the value.
            return min;
        }

        long[] hist = new long[OTSU_BINS];
        double scale = OTSU_BINS / range;
        for (double v : sorted) {
            int bin = (int) ((v - min) * scale);
            if (bin < 0) {
                bin = 0;
            } else if (bin >= OTSU_BINS) {
                bin = OTSU_BINS - 1;
            }
            hist[bin]++;
        }

        double total = n;
        double sumAll = 0.0;
        for (int b = 0; b < OTSU_BINS; b++) {
            sumAll += (b + 0.5) * hist[b];
        }

        double sumBackground = 0.0;
        double weightBackground = 0.0;
        double maxBetween = -1.0;
        int bestBin = 0;
        for (int b = 0; b < OTSU_BINS; b++) {
            weightBackground += hist[b];
            if (weightBackground == 0.0) {
                continue;
            }
            double weightForeground = total - weightBackground;
            if (weightForeground == 0.0) {
                break;
            }
            sumBackground += (b + 0.5) * hist[b];
            double meanBackground = sumBackground / weightBackground;
            double meanForeground = (sumAll - sumBackground) / weightForeground;
            double diff = meanBackground - meanForeground;
            double between = weightBackground * weightForeground * diff * diff;
            if (between > maxBetween) {
                maxBetween = between;
                bestBin = b;
            }
        }

        // Threshold sits at the upper edge of the best background bin.
        return min + ((bestBin + 1.0) / OTSU_BINS) * range;
    }

    /**
     * Count values strictly below {@code threshold} in an ascending-sorted array
     * using binary search.
     */
    private static long countBelow(double[] sorted, double threshold) {
        if (Double.isNaN(threshold)) {
            return 0L;
        }
        int idx = Arrays.binarySearch(sorted, threshold);
        if (idx >= 0) {
            // Walk back to the first occurrence so we count strictly-below only.
            while (idx > 0 && sorted[idx - 1] == threshold) {
                idx--;
            }
            return idx;
        }
        // Negative: idx = -(insertionPoint) - 1; insertion point == count below.
        return -(idx) - 1;
    }
}
