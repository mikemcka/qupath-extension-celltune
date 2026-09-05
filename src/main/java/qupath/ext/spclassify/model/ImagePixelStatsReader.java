package qupath.ext.spclassify.model;

import java.awt.image.BufferedImage;
import java.awt.image.DataBuffer;
import java.awt.image.Raster;
import java.awt.image.SampleModel;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import qupath.lib.images.servers.ImageServer;
import qupath.lib.regions.RegionRequest;

/**
 * Reads a low-resolution version of a (typically pyramidal) image from an
 * {@link ImageServer} and computes whole-image {@link ImagePixelStats.ImageStats}.
 * <p>
 * Part of the image pixel prescreen, whose statistics methodology was adapted from
 * <a href="https://pypi.org/project/pixel-patrol/">pixel-patrol</a> (MIT) — see
 * {@link ImagePixelStats} for the attribution note.
 * <p>
 * The resolution is chosen so the long edge of the region read is approximately
 * {@link #DEFAULT_MAX_LONG_EDGE} pixels: the requested downsample is
 * {@code max(1, longEdge / maxLongEdge)}, and QuPath serves it from the nearest
 * pyramid level. Reading every image to (roughly) the same pixel footprint keeps
 * the cohort statistics comparable like-for-like regardless of each slide's
 * native size or pyramid structure.
 * <p>
 * <b>Memory note:</b> one downsampled region (all channels) is held in memory at
 * once, plus a transient per-channel copy. At the default 2048px target this is
 * modest for a handful of channels but grows with channel count; lower
 * {@code maxLongEdge} for very highly multiplexed images.
 */
public final class ImagePixelStatsReader {

    /** Default target for the long edge of the low-resolution image, in pixels. */
    public static final int DEFAULT_MAX_LONG_EDGE = 2048;

    private ImagePixelStatsReader() {}

    /**
     * Read and analyse one image at the default long-edge target.
     *
     * @param imageName display name to record on the result
     * @param server    the image server (plane z=0, t=0 is read)
     * @return whole-image statistics
     * @throws IOException if the region cannot be read
     */
    public static ImagePixelStats.ImageStats read(String imageName, ImageServer<BufferedImage> server)
            throws IOException {
        return read(imageName, server, DEFAULT_MAX_LONG_EDGE);
    }

    /**
     * Read and analyse one image at a given long-edge target.
     *
     * @param imageName   display name to record on the result
     * @param server      the image server (plane z=0, t=0 is read)
     * @param maxLongEdge target for the long edge of the region read, in pixels
     * @return whole-image statistics
     * @throws IOException if the region cannot be read
     */
    public static ImagePixelStats.ImageStats read(String imageName, ImageServer<BufferedImage> server, int maxLongEdge)
            throws IOException {

        int fullWidth = server.getWidth();
        int fullHeight = server.getHeight();
        int longEdge = Math.max(fullWidth, fullHeight);
        double downsample = Math.max(1.0, (double) longEdge / Math.max(1, maxLongEdge));

        RegionRequest request = RegionRequest.createInstance(server.getPath(), downsample, 0, 0, fullWidth, fullHeight);
        BufferedImage img = server.readRegion(request);
        if (img == null) {
            throw new IOException("Image server returned no pixels for " + imageName);
        }

        Raster raster = img.getRaster();
        int w = raster.getWidth();
        int h = raster.getHeight();
        int nBands = raster.getNumBands();
        double dtypeMax = pixelTypeMax(raster.getSampleModel());

        List<String> channelNames = channelNames(server, nBands);

        // Extract one float[] per channel, then release the BufferedImage before
        // the (transient) per-channel arrays peak alongside the statistic copies.
        float[][] channelValues = new float[nBands][];
        for (int b = 0; b < nBands; b++) {
            channelValues[b] = raster.getSamples(0, 0, w, h, b, (float[]) null);
        }
        img = null;
        raster = null;

        return ImagePixelStats.compute(imageName, downsample, w, h, channelNames, channelValues, dtypeMax);
    }

    /**
     * Resolve channel display names from the server metadata, falling back to
     * generic labels when metadata is missing or shorter than {@code nBands}.
     */
    private static List<String> channelNames(ImageServer<BufferedImage> server, int nBands) {
        var names = new ArrayList<String>(nBands);
        List<qupath.lib.images.servers.ImageChannel> channels = null;
        try {
            channels = server.getMetadata().getChannels();
        } catch (Exception ignored) {
            // Fall back to generic names below.
        }
        for (int b = 0; b < nBands; b++) {
            String name = null;
            if (channels != null && b < channels.size() && channels.get(b) != null) {
                name = channels.get(b).getName();
            }
            names.add((name == null || name.isBlank()) ? ("Channel " + (b + 1)) : name);
        }
        return names;
    }

    /**
     * Maximum representable value for the raster's storage type, or {@code NaN}
     * for floating-point rasters where saturation is undefined.
     */
    private static double pixelTypeMax(SampleModel sampleModel) {
        int dataType = sampleModel.getDataType();
        if (dataType == DataBuffer.TYPE_FLOAT || dataType == DataBuffer.TYPE_DOUBLE) {
            return Double.NaN;
        }
        int bits = sampleModel.getSampleSize(0);
        if (bits <= 0 || bits >= 53) {
            return Double.NaN;
        }
        return Math.pow(2.0, bits) - 1.0;
    }
}
