// consoleAp.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#define NOMINMAX

#include <iostream>
#include <set>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>

#define _CRT_SECURE_NO_WARNINGS

#define M_PI                3.14159265358979323846  // pi

std::wstring AppendWstr( const std::wstring& str1, const std::wstring& str2 )
{
    std::wstring str = str1 + str2;
    return str;
}

void AppendWstr2( __inout std::wstring& str1, const std::wstring& str2 )
{
    str1 += str2;
}

void AppendWstr3( __inout std::wstring& str1, std::wstring str2 )
{
    str1 += str2;
}

static size_t s_nTotalSize = 0;

void* operator new(size_t size)
{
    printf("new called %lld\n", size);
    s_nTotalSize += size;
    return malloc(size);
}



// Custom float3 class to replace vec3
class float3 {
public:
    float x, y, z;

    // Constructors
    float3() : x( 0.0f ), y( 0.0f ), z( 0.0f ) {}
    float3( float val ) : x( val ), y( val ), z( val ) {}
    float3( float x, float y, float z ) : x( x ), y( y ), z( z ) {}

    // Basic arithmetic operators
    float3 operator+( const float3& other ) const {
        return float3( x + other.x, y + other.y, z + other.z );
    }

    float3 operator-( const float3& other ) const {
        return float3( x - other.x, y - other.y, z - other.z );
    }

    float3 operator*( float other ) const {
        return float3( x * other, y * other, z * other );
    }

    float3 operator*( const float3& other ) const {
        return float3( x * other.x, y * other.y, z * other.z );
    }

    float3 operator/( const float3& other ) const {
        return float3( x / other.x, y / other.y, z / other.z );
    }

    float3 operator/( float scalar ) const {
        return float3( x / scalar, y / scalar, z / scalar );
    }

    // Dot product
    float dot( const float3& other ) const {
        return x * other.x + y * other.y + z * other.z;
    }
    float length() const {
        return std::sqrt( x * x + y * y + z * z );
    }
};

float GetSearchLightDiffuseScalingFactor( float SurfaceAlbedo )
{
    return 3.5 + 100 * std::pow( SurfaceAlbedo - 0.33, 4 );
}

float3 GetSearchLightDiffuseScalingFactor( float3 SurfaceAlbedo )
{
    // ...
    float3 ret;
    ret.x = GetSearchLightDiffuseScalingFactor( SurfaceAlbedo.x );
    ret.y = GetSearchLightDiffuseScalingFactor( SurfaceAlbedo.y );
    ret.z = GetSearchLightDiffuseScalingFactor( SurfaceAlbedo.z );
    return ret;
}

float GetPerpendicularScalingFactor( float SurfaceAlbedo )
{
    // add abs() to match the formula in the original paper. 
    return 1.85 - SurfaceAlbedo + 7 * std::pow( std::fabs( SurfaceAlbedo - 0.8 ), 3 );
}

float3 GetPerpendicularScalingFactor( float3 SurfaceAlbedo )
{
    // ...
    float3 ret;
    ret.x = GetPerpendicularScalingFactor( SurfaceAlbedo.x );
    ret.y = GetPerpendicularScalingFactor( SurfaceAlbedo.y );
    ret.z = GetPerpendicularScalingFactor( SurfaceAlbedo.z );
    return ret;
}

float GetDiffuseMeanFreePathFromMeanFreePath( float SurfaceAlbedo, float MeanFreePath )
{
    return MeanFreePath * GetSearchLightDiffuseScalingFactor( SurfaceAlbedo ) / GetPerpendicularScalingFactor( SurfaceAlbedo );
}

float3 GetDiffuseMeanFreePathFromMeanFreePath( float3 SurfaceAlbedo, float3 MeanFreePath )
{
    return MeanFreePath * GetSearchLightDiffuseScalingFactor( SurfaceAlbedo ) / GetPerpendicularScalingFactor( SurfaceAlbedo );
}

float sss_diffusion_profile_scatterDistance( const float surfaceAlbedo ) {
    const float a = surfaceAlbedo - float( 0.8 );
    return 1.9 - surfaceAlbedo + 3.5 * a * a;
}


float3 normalize( const float3 in )
{
    float len = std::sqrt( in.x * in.x + in.y * in.y + in.z * in.z );
    return float3( in.x / len, in.y / len, in.z / len );
}

float3 sss_diffusion_profile_evaluate( float radius, const float3& scatterDistance ) {
    // Prevent division by zero
    const float epsilon = 0.000001f;
    float3 safeScatterDistance(
        std::max( scatterDistance.x, epsilon ),
        std::max( scatterDistance.y, epsilon ),
        std::max( scatterDistance.z, epsilon )
    );

    if ( radius <= 0 ) {
        // Early return for zero or negative radius
        return ( float3( 0.25f ) / static_cast<float>( M_PI ) ) / safeScatterDistance;
    }

    // Calculate reduced distance
    const float3 rd = float3( radius ) / safeScatterDistance;

    // Exponential terms
    const float3 exp1 = float3( std::exp( -rd.x ), std::exp( -rd.y ), std::exp( -rd.z ) );
    const float3 exp2 = float3(
        std::exp( -rd.x / 3.0f ),
        std::exp( -rd.y / 3.0f ),
        std::exp( -rd.z / 3.0f )
    );

    // Denominator calculation
    const float3 denominator = float3( 8.0f * static_cast<float>( M_PI ) ) * safeScatterDistance * float3( radius );

    // Combine terms
    return ( exp1 + exp2 ) / denominator;
}

class DisneyBSSRDF {
public:
    // Helper function to mimic GLSL's clamp
    static float clamp( float val, float min, float max ) {
        return std::max( min, std::min( val, max ) );
    }

    // Disney Schlick Weight function
    static float disney_schlickWeight( float a ) {
        const float b = clamp( 1.0f - a, 0.0f, 1.0f );
        const float bb = b * b;
        return bb * bb * b;
    }

    // Disney Diffuse Lambert Weight (with two inputs)
    static float disney_diffuseLambertWeight( float fv, float fl ) {
        return ( 1.0f - 0.5f * fl ) * ( 1.0f - 0.5f * fv );
    }

    // Disney Diffuse Lambert Weight (single input)
    static float disney_diffuseLambertWeightSingle( float f ) {
        return 1.0f - 0.5f * f;
    }

    // Subsurface Scattering Fresnel Evaluate
    static float3 disney_bssrdf_fresnel_evaluate( const float3& normal, const float3& direction ) {
        const float dotND = normal.dot( direction );
        const float schlick = disney_schlickWeight( dotND );
        const float lambertWeight = disney_diffuseLambertWeightSingle( schlick );
        return float3( lambertWeight );
    }

    // Subsurface Scattering Evaluate (first overload)
    static void disney_bssrdf_evaluate(
        const float3& normal,
        const float3& v,
        float distance,
        const float3& scatterDistance,
        const float3& surfaceAlbedo,
        float3& bssrdf
    ) {
        
        const float3 diffusionProfile = surfaceAlbedo * sss_diffusion_profile_evaluate( distance, scatterDistance );
        bssrdf = diffusionProfile / static_cast<float>( M_PI );
    }

    // Subsurface Scattering Evaluate (second overload)
    static void disney_bssrdf_evaluate(
        const float3& normal,
        const float3& v,
        const float3& normalSample,
        const float3& l,
        float distance,
        const float3& scatterDistance,
        const float3& surfaceAlbedo,
        float3& bssrdf,
        float3& bsdf
    ) {
        
        const float3 diffusionProfile = surfaceAlbedo * sss_diffusion_profile_evaluate( distance, scatterDistance );
        bssrdf = diffusionProfile / static_cast<float>( M_PI );
        bssrdf = bssrdf * disney_bssrdf_fresnel_evaluate( normal, v );
        bsdf = disney_bssrdf_fresnel_evaluate( normalSample, l ); //float3( 1.0f, 1.0f, 1.0f );
    }

};

// Forward declarations for undefined types
struct BSDFFrame {
    float3 n;  // normal
    float3 t;  // tangent
    float3 b;  // bitangent
};


constexpr float SSS_SAMPLING_DISK_AXIS_0_WEIGHT = 0.5f;
constexpr float SSS_SAMPLING_DISK_AXIS_1_WEIGHT = 0.25f;
constexpr float SSS_SAMPLING_DISK_AXIS_2_WEIGHT = 0.25f;
constexpr float SSS_SAMPLING_DISK_CHANNEL_0_WEIGHT = 0.33f;
constexpr float SSS_SAMPLING_DISK_CHANNEL_1_WEIGHT = 0.33f;
constexpr float SSS_SAMPLING_DISK_CHANNEL_2_WEIGHT = 0.33f;


float3 sss_diffusion_profile_pdf_vectorized( float proj, const float3& scatterDistance ) {
    float3 pdf = sss_diffusion_profile_evaluate( proj, scatterDistance );
    return pdf;
}

float sss_sampling_disk_pdf(
    const float3 distanceVector,
    //const float3& position,
    const BSDFFrame& frame,
    //const float3& samplePosition,
    const float3& sampleNormal,
    const float3& scatterDistance
) {
    // Calculate distance vector
    const float3 d = distanceVector; // samplePosition - position;

    // Project distance to local frame
    const float3 dLocal = float3(
        frame.n.dot( d ),
        frame.t.dot( d ),
        frame.b.dot( d )
    );

    // Calculate projected radii
    const float3 rProj = float3(
        std::sqrt( dLocal.y * dLocal.y + dLocal.z * dLocal.z ),
        std::sqrt( dLocal.z * dLocal.z + dLocal.x * dLocal.x ),
        std::sqrt( dLocal.x * dLocal.x + dLocal.y * dLocal.y )
    );

    // Project normal to local frame
    const float3 nLocal = float3(
        std::abs( frame.n.dot( sampleNormal ) ),
        std::abs( frame.t.dot( sampleNormal ) ),
        std::abs( frame.b.dot( sampleNormal ) )
    );

    // Define axis and channel probabilities
    const float3 axisProb = float3(
        SSS_SAMPLING_DISK_AXIS_0_WEIGHT,
        SSS_SAMPLING_DISK_AXIS_1_WEIGHT,
        SSS_SAMPLING_DISK_AXIS_2_WEIGHT
    );

    const float3 channelProb = float3(
        SSS_SAMPLING_DISK_CHANNEL_0_WEIGHT,
        SSS_SAMPLING_DISK_CHANNEL_1_WEIGHT,
        SSS_SAMPLING_DISK_CHANNEL_2_WEIGHT
    );

    float pdf = 0.0f;

    // Compute PDF for each axis and channel
    float3 pdfAxis = sss_diffusion_profile_pdf_vectorized( rProj.x, scatterDistance )
        * axisProb.x * channelProb.x * nLocal.x;
    pdf += pdfAxis.x + pdfAxis.y + pdfAxis.z;

    pdfAxis = sss_diffusion_profile_pdf_vectorized( rProj.y, scatterDistance )
        * axisProb.y * channelProb.y * nLocal.y;
    pdf += pdfAxis.x + pdfAxis.y + pdfAxis.z;

    pdfAxis = sss_diffusion_profile_pdf_vectorized( rProj.z, scatterDistance )
        * axisProb.z * channelProb.z * nLocal.z;
    pdf += pdfAxis.x + pdfAxis.y + pdfAxis.z;

    return pdf;
}
const float Dmfp2MfpMagicNumber = 0.6f;

class EBuiltinShadername {
public:
    static constexpr const char* RLHEAD_SHADER = "RLHead";
};


float3 reflect( float3 i, float3 n )
{
    //float3 n2 = n;
    //float dotValue = n.dot( i ) * 2;
    //n2 = n2 * dotValue;
    //float3 ret = i - n2;
    
    return i - n * 2.0f * n.dot( i );
}

float dot( float3 a, float3 b )
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

float3 AdjustShadingNormal( float3 ShadingNormal, float3 GeoNormal, float3 RayDirection )
{
    float3 D = RayDirection;
    float3 R = reflect( D, ShadingNormal );
    float k = dot( R, GeoNormal );
    float3 kGeoNormal = GeoNormal * k;
    if ( k < 0.0 )
    {
        return normalize( normalize( R - kGeoNormal ) - D );
    }
    return ShadingNormal;
}


float3 abs( float3 in )
{
    return float3( std::abs( in.x ), std::abs( in.y ), std::abs( in.z ) );
}

float3 GetPerpendicularScalingFactor3D( float3 SurfaceAlbedo )
{
    float3 Value = abs( SurfaceAlbedo - float3(0.8) );
    return float3(1.85) - SurfaceAlbedo + float3(7) * Value * Value * Value;
}

float3 GetSearchLightDiffuseScalingFactor3D( float3 SurfaceAlbedo )
{
    float3 Value = SurfaceAlbedo - float3(0.33);
    return float3(3.5) + float3(100) * Value * Value * Value * Value;
}

float3 GetMFPFromDMFPCoeff( float3 DMFPSurfaceAlbedo, float3 MFPSurfaceAlbedo, float Dmfp2MfpMagicNumber = 0.6f )
{
    return float3(Dmfp2MfpMagicNumber) * GetPerpendicularScalingFactor3D( MFPSurfaceAlbedo ) / GetSearchLightDiffuseScalingFactor3D( DMFPSurfaceAlbedo );
}

float3 GetMFPFromDMFPApprox( float3 SurfaceAlbedo, float3 TargetSurfaceAlbedo, float3 DMFP )
{
    return GetMFPFromDMFPCoeff( SurfaceAlbedo, TargetSurfaceAlbedo ) * DMFP;
}

constexpr float BURLEY_CM_2_MM = 10.0f;
constexpr float BURLEY_MM_2_CM = 0.1f;

float3 DecodeDiffuseMeanFreePath( float3 EncodedDiffuseMeanFreePath )
{
    return EncodedDiffuseMeanFreePath * 1 / ( 0.01f * 0.2f );
}

float DecodeWorldUnitScale( float EncodedWorldUnitScale )
{
    return EncodedWorldUnitScale * 1 / 0.02f;
}
float EncodeWorldUnitScale( float WorldUnitScale )
{
    return WorldUnitScale * 0.02f; // ENC_WORLDUNITSCALE_IN_CM_TO_UNIT;
}
float3 DecodeSSSProfileRadiusV0( float3 SurfaceAlbedo, float EncodedWorldUnitScale, float3 EncodedDiffuseMeanFreePath, float3 DiffuseColor, float Opacity )
{
    // Burley parameterization

    //SurfaceAlbedo = { R = 0.906040013 G = 0.335260004 B = 0.271840006 }

    //{ R = 5.38478756 G = 0.479125202 B = 0.388162941  A = R = 5.3 }
    float3 DiffuseMeanFreePath = DecodeDiffuseMeanFreePath( EncodedDiffuseMeanFreePath );
    float WorldUnitScale = DecodeWorldUnitScale( EncodedWorldUnitScale );

    // Opacity acts as a per-pixel radius multiplier
    // NOTE: this seems backwards? Opacity=0 acts like default-lit while Opacity=1 acts like SSS?
    // NOTE2: Confirm if this interpretation of opacity is correct ...
    float3 SSSRadius = GetMFPFromDMFPApprox( SurfaceAlbedo, DiffuseColor, float3(Opacity * WorldUnitScale) * DiffuseMeanFreePath );

    return SSSRadius * BURLEY_MM_2_CM;
}

float3 DecodeSSSProfileRadiusV1( float3 SurfaceAlbedo, float WorldUnitScale, float3 EncodedDiffuseMeanFreePath, float3 DiffuseColor, float Opacity )
{
    // Burley parameterization

    //SurfaceAlbedo = { R = 0.906040013 G = 0.335260004 B = 0.271840006 }

    //{ R = 5.38478756 G = 0.479125202 B = 0.388162941  A = R = 5.3 }
    float3 DiffuseMeanFreePath = DecodeDiffuseMeanFreePath( EncodedDiffuseMeanFreePath );

    // Opacity acts as a per-pixel radius multiplier
    // NOTE: this seems backwards? Opacity=0 acts like default-lit while Opacity=1 acts like SSS?
    // NOTE2: Confirm if this interpretation of opacity is correct ...
    float3 SSSRadius = GetMFPFromDMFPApprox( SurfaceAlbedo, DiffuseColor, float3( Opacity * WorldUnitScale ) * DiffuseMeanFreePath );

    return SSSRadius * BURLEY_MM_2_CM;
}

float3 DecodeSSSProfileRadius( float3 SurfaceAlbedo, float WorldUnitScale, float3 DiffuseMeanFreePath, float3 DiffuseColor, float Opacity )
{
    // Burley parameterization

    //SurfaceAlbedo = { R = 0.906040013 G = 0.335260004 B = 0.271840006 }

    //{ R = 5.38478756 G = 0.479125202 B = 0.388162941  A = R = 5.3 }

    // Opacity acts as a per-pixel radius multiplier
    // NOTE: this seems backwards? Opacity=0 acts like default-lit while Opacity=1 acts like SSS?
    // NOTE2: Confirm if this interpretation of opacity is correct ...
    float3 SSSRadius = GetMFPFromDMFPApprox( SurfaceAlbedo, DiffuseColor, float3( Opacity * WorldUnitScale ) * DiffuseMeanFreePath );

    return SSSRadius * BURLEY_MM_2_CM;
}

#define ENC_DIFFUSEMEANFREEPATH_IN_MM_TO_UNIT (0.01f*0.2f)
#define DEC_UNIT_TO_DIFFUSEMEANFREEPATH_IN_MM 1/ENC_DIFFUSEMEANFREEPATH_IN_MM_TO_UNIT

float3 EncodeDiffuseMeanFreePath( float3 DiffuseMeanFreePath )
{
    return DiffuseMeanFreePath * ENC_DIFFUSEMEANFREEPATH_IN_MM_TO_UNIT;
}

float saturate( float in )
{
    if ( in < 0 )
        in = 0;
    if ( in > 1 )
    {
        in = 1;
    }
    return in;
}

float ShadowTerminatorTerm( float3 L, float3 N, float3 Ns )
{
    const float Epsilon = 1e-6;
    const float CosD = saturate( abs( dot( Ns, N ) ) );
    const float Tan2D = ( 1.0 - CosD * CosD ) / ( CosD * CosD + Epsilon );
    const float Alpha2 = saturate( 0.125 * Tan2D );
    const float CosI = saturate( dot( Ns, L ) );
    const float Tan2I = ( 1.0f - CosI * CosI ) / ( CosI * CosI + Epsilon );
    return CosI > 0 ? 2.0f / ( 1.0f + sqrt( 1.0f + Alpha2 * Tan2I ) ) : 0.0;
    //#line 210 "/Engine/Private/PathTracing/Material/PathTracingMaterialCommon.ush"
}

#define LOG2_E 1.44269504089
//// https://zero-radiance.github.io/post/sampling-diffusion/
//// Performs sampling of a Normalized Burley diffusion profile in polar coordinates.
//// 'u' is the random number (the value of the CDF): [0, 1).
//// rcp(s) = 1 / ShapeParam = ScatteringDistance.
//// 'r' is the sampled radial distance, s.t. (u = 0 -> r = 0) and (u = 1 -> r = Inf).
//// rcp(Pdf) is the reciprocal of the corresponding PDF value.
float sampleBurleyDiffusionProfileAnalytical( float u, const float rcpS ) {
    u = 1 - u;// Convert CDF to CCDF; the resulting value of (u != 0)

    const float g = 1 + ( 4 * u ) * ( 2 * u + sqrt( 1 + ( 4 * u ) * u ) );
    const float n = exp2( log2( g ) * ( -1.0 / 3.0 ) );// g^(-1/3)
    const float p = ( g * n ) * n;// g^(+1/3)
    const float c = 1 + p + n;// 1 + g^(+1/3) + g^(-1/3)
    const float x = ( 3 / LOG2_E ) * log2( c / ( 4 * u ) );// 3 * Log[c / (4 * u)]

    // x      = s * r
    // exp_13 = Exp[-x/3] = Exp[-1/3 * 3 * Log[c / (4 * u)]]
    // exp_13 = Exp[-Log[c / (4 * u)]] = (4 * u) / c
    // exp_1  = Exp[-x] = exp_13 * exp_13 * exp_13
    // expSum = exp_1 + exp_13 = exp_13 * (1 + exp_13 * exp_13)
    // rcpExp = rcp(expSum) = c^3 / ((4 * u) * (c^2 + 16 * u^2))
    const float rcpExp = ( ( c * c ) * c ) / ( ( 4 * u ) * ( ( c * c ) + ( 4 * u ) * ( 4 * u ) ) );

    return x * rcpS; // r
}

float sss_diffusion_profile_sample( const float xi, const float scatterDistance ) {
    return sampleBurleyDiffusionProfileAnalytical( xi, scatterDistance );
}

void TestScatterDistance()
{
    float3 skinColor( 0.9, 0.3, 0.3 );
    float3 MeanFreePathColor( 0.5, .5, 0.5 );

    float3 scatterDistance = MeanFreePathColor / GetPerpendicularScalingFactor3D( skinColor );
    printf("sss distance:\n%g\n, %g\n, %g\n", scatterDistance.x, scatterDistance.y, scatterDistance.z );

    bool useLength = true;
    const float sampledScatterDistance = useLength ? scatterDistance.length() : scatterDistance.y;
    const float radius = sss_diffusion_profile_sample( 0.5, sampledScatterDistance );
    printf("radius %g\n", radius);
    const float radiusMax = sss_diffusion_profile_sample( 0.999, sampledScatterDistance );
    printf( "radius MAX %g\n", radiusMax );
}

int main( int argc, char**argv )
{
    TestScatterDistance();
    const float SSSGuidingRatio = 0.5;
    float G = 0;

    const float GuidedRatio = SSSGuidingRatio * ( 1.0 - pow( abs( G * 4 ), 0.0625 ) );

    //Data.SurfaceAlbedo = {R=0.00899999961 G=1.00000000 B=0.00899999961 ...}
    float3 DiffuseColor0( 0.5, .5, 0.5 );
    float3 SurfaceAlbedo( 0.0089, 1, 0.0089 );
    float3 MeanFreePathColor( 0.2, 0.2, 0.2 );
    float MeanFreePathDistance = 1; // in cm
    constexpr float CmToMm = 10.f;

    float ssWeight = 1;
    float WorldUnitScale = 1;

    float3 DiffuseMeanFreePathInMm = GetDiffuseMeanFreePathFromMeanFreePath( SurfaceAlbedo, MeanFreePathColor * MeanFreePathDistance ) * CmToMm / Dmfp2MfpMagicNumber;
    float3 SSSRadius = GetMFPFromDMFPApprox( SurfaceAlbedo, DiffuseColor0, DiffuseMeanFreePathInMm );
    SSSRadius = SSSRadius * 0.1;
    //{R=2.86610389 G=87.0166321 B=2.86610389 ...}


    float3 encodedDiffuseMeanFreaPath = EncodeDiffuseMeanFreePath( DiffuseMeanFreePathInMm );

    float3 skinColor( 0.9, 0.3, 0.3 );

     SSSRadius = DecodeSSSProfileRadius( SurfaceAlbedo, WorldUnitScale, DiffuseMeanFreePathInMm, skinColor, 1.0 );

    //{ R = 0.477683991 G = 0.174033269 B = 1.00000000 ... }

    //float3 SSSRadius = DecodeSSSProfileRadiusV1( SurfaceAlbedo, WorldUnitScale, encodedDiffuseMeanFreaPath, SurfaceAlbedo, 1.0 );

    // {x=0.00733244745 y=0.00607479457 z=0.00579881435 }

    float3 adjN= AdjustShadingNormal( float3( 0, 0, -1 ), float3( 0, 0, -1 ), float3( 1, 0, -1 ) );

    float scatterDistance2 = 1 * 1 / sss_diffusion_profile_scatterDistance( 0.33 );

    // SurfaceAlbedo 0.9, 0.33, 0.27
    //MeanFreePathColor {0.21, 0.18, 0.17, 1.00000000}
    // 0.1 cm

    float dmfpCm = GetDiffuseMeanFreePathFromMeanFreePath( 1, 0.1 );
    float dmfp = dmfpCm * 10 / Dmfp2MfpMagicNumber;
    //{ R = 16.4293327 G = 0.00479125232 B = 0.107140414 ... }
    // +		GetDiffuseMeanFreePathFromMeanFreePath returned	{R=0.32 G=0.028 B=0.023 ...}	FLinearColor
    //  * 60 
    // dmfp in mm, {R=5.3 G=0.479125202 B=0.388162941 ...}

    // Out.DiffuseMeanFreePath = DecodeDiffuseMeanFreePath(GetSubsurfaceProfileDiffuseMeanFreePath(SubsurfaceProfileInt));

    std::cout << "albedo: 0.33, mean free path: 1.0, dmfp = " << dmfp << std::endl;
    std::cout << "scatterDistance: " << scatterDistance2 << std::endl;

    const float3 normal( 0, 0, 1 );
    float3 v( 1, 0, 1 );
    v = normalize( v );
    const float3 normalSample = normal;
    float3 l = normal;
    float distance = 0;
    float3 scatterDistance( 0.98, 0.18, 0.08 );
    float3 surfaceAlbedo( 0.5, 0.5, 0.5 );
    float3 bssrdf( 0 );
    float3 bsdf( 0 );

    BSDFFrame frame;
    frame.n = normal;
    frame.t = float3( 1, 0, 0 );
    frame.b = float3( 0, -1, 0 );
    DisneyBSSRDF::disney_bssrdf_evaluate( normal, v, normalSample, l, distance, scatterDistance, surfaceAlbedo, bssrdf, bsdf );

    float3 bssrdfPDF = sss_sampling_disk_pdf( float3(0.1), frame, normal, scatterDistance );

    std::cout << "bssrdfPDF:" << bssrdfPDF.x << " " << bssrdfPDF.y << " " << bssrdfPDF.z << std::endl;
    std::cout << "bssrdf:" << bssrdf.x << " " << bssrdf.y << " " << bssrdf.z << std::endl;
    std::cout << "bsdf:" << bsdf.x << " " << bsdf.y << " " << bsdf.z << std::endl;

    return 0;

    std::wstring str1 = L"12345678901234567890";
    std::wstring str2 = L"RLRLabcdefghijklmn_opqrstuvwxzy";

    //std::wstring str3 = AppendWstr(str1, str2); //48
    std::wstring str3 = str1 + str2; //224
    std::wcout << str3 << std::endl;

    //AppendWstr2( str1, str2 ); // 224
    //AppendWstr3(str1, str2); // 288
    std::wcout << str1 << std::endl;
    printf("%lld\n", s_nTotalSize );
    return 0;
}


