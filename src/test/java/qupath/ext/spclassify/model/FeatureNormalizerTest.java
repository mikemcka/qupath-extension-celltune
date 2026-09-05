package qupath.ext.spclassify.model;

import static org.junit.jupiter.api.Assertions.*;

import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class FeatureNormalizerTest {

    private static final double DELTA = 1e-5;

    @Test
    void applyNoneReturnsRawValue() {
        var fn = new FeatureNormalizer();
        assertEquals(3.5f, fn.apply("CD3", 3.5f), (float) DELTA);
    }

    @Test
    void applyArcsinhTransformsCorrectly() {
        var fn = new FeatureNormalizer();
        fn.setTransform("CD3", FeatureNormalizer.Transform.ARCSINH);
        // arcsinh(1/1) = ln(1 + sqrt(2)) ≈ 0.88137
        float result = fn.apply("CD3", 1.0f);
        assertEquals(Math.log(1 + Math.sqrt(2)), result, DELTA);
    }

    @Test
    void applyArcsinhUsesConfiguredCofactor() {
        var fn = new FeatureNormalizer();
        fn.setTransform("CD4", FeatureNormalizer.Transform.ARCSINH);
        fn.setArcsinhCofactor(5.0);
        // arcsinh(5/5) = arcsinh(1) ≈ 0.88137
        float result = fn.apply("CD4", 5.0f);
        assertEquals(Math.log(1 + Math.sqrt(2)), result, DELTA);
    }

    @Test
    void applySqrtTransformsCorrectly() {
        var fn = new FeatureNormalizer();
        fn.setTransform("CD8", FeatureNormalizer.Transform.SQRT);
        assertEquals(2.0f, fn.apply("CD8", 4.0f), (float) DELTA);
    }

    @Test
    void applySqrtClampsNegativeValueToZero() {
        var fn = new FeatureNormalizer();
        fn.setTransform("CD8", FeatureNormalizer.Transform.SQRT);
        assertEquals(0.0f, fn.apply("CD8", -9.0f), (float) DELTA);
    }

    @Test
    void setArcsinhCofactorThrowsOnNonPositive() {
        var fn = new FeatureNormalizer();
        assertThrows(IllegalArgumentException.class, () -> fn.setArcsinhCofactor(0.0));
        assertThrows(IllegalArgumentException.class, () -> fn.setArcsinhCofactor(-1.0));
    }

    @Test
    void setTransformNoneRemovesTransform() {
        var fn = new FeatureNormalizer();
        fn.setTransform("CD3", FeatureNormalizer.Transform.ARCSINH);
        fn.setTransform("CD3", FeatureNormalizer.Transform.NONE);
        assertEquals(FeatureNormalizer.Transform.NONE, fn.getTransform("CD3"));
        assertFalse(fn.hasTransforms());
    }

    @Test
    void setTransformOnCollectionAppliesAll() {
        var fn = new FeatureNormalizer();
        fn.setTransform(List.of("CD3", "CD4"), FeatureNormalizer.Transform.SQRT);
        assertEquals(FeatureNormalizer.Transform.SQRT, fn.getTransform("CD3"));
        assertEquals(FeatureNormalizer.Transform.SQRT, fn.getTransform("CD4"));
    }

    @Test
    void hasTransformsReturnsFalseWhenEmpty() {
        var fn = new FeatureNormalizer();
        assertFalse(fn.hasTransforms());
    }

    @Test
    void hasTransformsReturnsTrueAfterSetting() {
        var fn = new FeatureNormalizer();
        fn.setTransform("CD3", FeatureNormalizer.Transform.ARCSINH);
        assertTrue(fn.hasTransforms());
    }

    @Test
    void toTransformMapOmitsNoneTransforms() {
        var fn = new FeatureNormalizer();
        fn.setTransform("CD3", FeatureNormalizer.Transform.ARCSINH);
        fn.setTransform("CD4", FeatureNormalizer.Transform.NONE);

        Map<String, String> map = fn.toTransformMap();
        assertTrue(map.containsKey("CD3"));
        assertFalse(map.containsKey("CD4"));
        assertEquals("ARCSINH", map.get("CD3"));
    }

    @Test
    void fromTransformMapRoundTrip() {
        var fn = new FeatureNormalizer();
        fn.setTransform("CD3", FeatureNormalizer.Transform.ARCSINH);
        fn.setTransform("CD8", FeatureNormalizer.Transform.SQRT);
        Map<String, String> map = fn.toTransformMap();

        var fn2 = new FeatureNormalizer();
        fn2.fromTransformMap(map);

        assertEquals(FeatureNormalizer.Transform.ARCSINH, fn2.getTransform("CD3"));
        assertEquals(FeatureNormalizer.Transform.SQRT, fn2.getTransform("CD8"));
    }

    @Test
    void fromTransformMapHandlesNullGracefully() {
        var fn = new FeatureNormalizer();
        fn.setTransform("CD3", FeatureNormalizer.Transform.ARCSINH);
        fn.fromTransformMap(null);
        assertFalse(fn.hasTransforms());
    }

    @Test
    void clearRemovesAllTransforms() {
        var fn = new FeatureNormalizer();
        fn.setTransform("CD3", FeatureNormalizer.Transform.ARCSINH);
        fn.setTransform("CD8", FeatureNormalizer.Transform.SQRT);
        fn.clear();
        assertFalse(fn.hasTransforms());
    }

    @Test
    void getTransformReturnsNoneForUnknownFeature() {
        var fn = new FeatureNormalizer();
        assertEquals(FeatureNormalizer.Transform.NONE, fn.getTransform("unknown"));
    }
}
