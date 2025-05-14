// consoleAp.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#define NOMINMAX

#include <iostream>

#include <windows.h>
#include <mmsystem.h>
#include <iostream>ls

#include <mmeapi.h>
#include <fstream>

#include <set>
#include <unordered_set>
#include <vector>
#include <chrono>
#include <random>
#pragma comment(lib, "winmm.lib")

int mainPlaySound()
{
    // Path to the WAV file
    const char* wavFilePath = "Mat183.wav"; // e:\\Testproj2024\\

    // Play the sound file
    if ( PlaySoundA( wavFilePath, NULL, SND_ASYNC ) ) {
        std::cout << "Sound played successfully!" << std::endl;
    }
    else {
        std::cerr << "Failed to play sound." << std::endl;
    }
    getchar();
    return 0;
}


// Function prototypes
bool LoadWAVFile( const char* filename, WAVEFORMATEX& wfx, BYTE*& data, DWORD& dataSize );
void PlayWavFile( const BYTE* audioData, UINT32 dataSize, WAVEFORMATEX& waveFormat );

void CleanupWAV( BYTE* data );



int TestInsertSetSpeed() {
    const int N = 5000;
    std::set<int> s;
    std::unordered_set<int> us;
    std::vector<int> v;
    std::vector<int> v2;



    // Seed with a real random value, if available
    std::random_device rd;

    // Choose a random number between 0 and N-1
    std::mt19937 gen( rd() );
    std::uniform_int_distribution<> dis( 0, N - 1 );

    for ( int i = 0; i < N; ++i ) {
        v.push_back( dis( gen ) );
    }

    // Measuring time for std::set
    auto start = std::chrono::high_resolution_clock::now();
    for ( int i = 0; i < N; ++i ) {
        s.insert( v[i] );
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "std::set insertion time: " << std::chrono::duration_cast<std::chrono::microseconds>( end - start ).count() << " us\n";

    // Measuring time for std::unordered_set
    start = std::chrono::high_resolution_clock::now();
    for ( int i = 0; i < N; ++i ) {
        us.insert( v[ i ] );
        //us.insert( dis( gen ) );
    }
    end = std::chrono::high_resolution_clock::now();
    std::cout << "std::unordered_set insertion time: " << std::chrono::duration_cast<std::chrono::microseconds>( end - start ).count() << " us\n";

    // Measuring time for std::vector
    start = std::chrono::high_resolution_clock::now();
    for ( int i = 0; i < N; ++i ) {
        v2.push_back( v[i] );
    }
    end = std::chrono::high_resolution_clock::now();
    std::cout << "std::vector insertion time: " << std::chrono::duration_cast<std::chrono::microseconds>( end - start ).count() << " us\n";

    return 0;
}

UINT numDevices = 0;

void QueryAudioDeviceCapabilities() {
    numDevices = waveOutGetNumDevs();
    std::cout << "Number of audio output devices: " << numDevices << std::endl;

    for ( UINT deviceID = 0; deviceID < numDevices; ++deviceID ) {
        WAVEOUTCAPS caps;
        MMRESULT result = waveOutGetDevCaps( deviceID, &caps, sizeof( WAVEOUTCAPS ) );

        if ( result == MMSYSERR_NOERROR ) {
            std::cout << "Device ID: " << deviceID << std::endl;
            std::cout << "  Manufacturer ID: " << caps.wMid << std::endl;
            std::cout << "  Product ID: " << caps.wPid << std::endl;
            std::cout << "  Driver version: " << ( caps.vDriverVersion >> 8 ) << "." << ( caps.vDriverVersion & 0xFF ) << std::endl;
            std::cout << "  Product name: " << caps.szPname << std::endl;
            std::cout << "  Supported formats: " << caps.dwFormats << std::endl;
            std::cout << "  Channels: " << caps.wChannels << std::endl;
        }
        else {
            std::cerr << "Error getting device capabilities for device ID: " << deviceID << std::endl;
        }
    }
}

int mainWaveout() {

    QueryAudioDeviceCapabilities();

    //return TestInsertSetSpeed();
    
    //const char* wavFilePath = "e:\\Testproj2024\\Mat183.wav";
    const char* wavFilePath = "Mat183.wav";
    WAVEFORMATEX wfx = { 0 };
    BYTE* data = nullptr;
    DWORD dataSize = 0;

    if ( !LoadWAVFile( wavFilePath, wfx, data, dataSize ) ) {
        std::cerr << "Failed to load WAV file." << std::endl;
        return 1;
    }
    std::cout << "Loaded WAV file: " << wavFilePath << std::endl;

    std::cout << "wfx.wFormatTag: " << wfx.wFormatTag << std::endl;
    std::cout << "wfx.nChannels: " << wfx.nChannels << std::endl;
    std::cout << "wfx.nSamplesPerSec: " << wfx.nSamplesPerSec << std::endl;
    std::cout << "wfx.nAvgBytesPerSec: " << wfx.nAvgBytesPerSec << std::endl;
    std::cout << "wfx.nBlockAlign: " << wfx.nBlockAlign << std::endl;
    std::cout << "wfx.wBitsPerSample: " << wfx.wBitsPerSample << std::endl;
    std::cout << "wfx.cbSize: " << wfx.cbSize << std::endl;

    HWAVEOUT hWaveOut = nullptr;
    MMRESULT result = MMSYSERR_NOERROR;
    std::cout << "-----------" << std::endl;
    std::cout << "Number of audio output devices: " << numDevices << std::endl;

    for ( int i = 0; i < numDevices; ++i )
    {
        std::cout << "Trying device ID: " << i << std::endl;
        result = waveOutOpen( &hWaveOut, WAVE_MAPPER, &wfx, 0, 0, CALLBACK_NULL );
        if ( result == MMSYSERR_NOERROR ) {
            std::cout << "Device ID: " << i << " OK!"<< std::endl;
            std::cout << "------------------------------------------------" << std::endl;
            break;
        }
        if ( result != MMSYSERR_NOERROR ) {
            std::cerr << "Failed to open waveform output device. result:" << result << std::endl;
            //CleanupWAV( data );
            //getchar();
            //return 1;
        }
    }

    if ( result != MMSYSERR_NOERROR ) {
        std::cerr << "Failed to open waveform output device. " << result << std::endl;
        CleanupWAV( data );
        getchar();
        return 1;
    }

    std::cout << "waveOutOpen() OK" << std::endl;

    WAVEHDR waveHeader = { 0 };
    waveHeader.lpData = reinterpret_cast<LPSTR>( data );
    waveHeader.dwBufferLength = dataSize;
    waveHeader.dwFlags = 0;

    result = waveOutPrepareHeader( hWaveOut, &waveHeader, sizeof( WAVEHDR ) );
    if ( result != MMSYSERR_NOERROR ) {
        std::cerr << "Failed to prepare waveform header." << std::endl;
        waveOutClose( hWaveOut );
        CleanupWAV( data );
        return 1;
    }
    std::cout << "waveOutPrepareHeader() OK" << std::endl;

    result = waveOutWrite( hWaveOut, &waveHeader, sizeof( WAVEHDR ) );
    if ( result != MMSYSERR_NOERROR ) {
        std::cerr << "Failed to write waveform data." << std::endl;
        waveOutUnprepareHeader( hWaveOut, &waveHeader, sizeof( WAVEHDR ) );
        waveOutClose( hWaveOut );
        CleanupWAV( data );
        return 1;
    }
    std::cout << "waveOutWrite() OK" << std::endl;

    std::cout << "Playing";
    // Wait for the sound to finish playing
    while ( !( waveHeader.dwFlags & WHDR_DONE ) ) {
        Sleep( 100 );
        std::cout << ".";
    }
    std::cout << "done" << std::endl;

    waveOutUnprepareHeader( hWaveOut, &waveHeader, sizeof( WAVEHDR ) );
    waveOutClose( hWaveOut );
    CleanupWAV( data );

    std::cout << "Sound played successfully!" << std::endl;
    return 0;
}


#include <windows.h>
#include <mmdeviceapi.h>
#include <audioclient.h>

#include <assert.h>
#define _USE_MATH_DEFINES
#include <math.h> // for sin()
#include <stdint.h>

// Struct to get data from loaded WAV file.
// NB: This will only work for WAV files containing PCM (non-compressed) data
// otherwise the layout will be different.
struct WavFile {
    // RIFF Chunk
    uint32_t riffId;
    uint32_t riffChunkSize;
    uint32_t waveId;

    // fmt Chunk
    uint32_t fmtId;
    uint32_t fmtChunkSize;
    uint16_t formatCode;
    uint16_t numChannels;
    uint32_t sampleRate;
    uint32_t byteRate;
    uint16_t blockAlign;
    uint16_t bitsPerSample;
    // These are not present for PCM Wav Files
    // uint16_t cbSize;
    // uint16_t wValidBitsPerSample;
    // uint32_t dwChannelMask;
    // char subFormatGUID[16];

    // data Chunk
    uint32_t dataId;
    uint32_t dataChunkSize;
    uint16_t samples; // actual samples start here
};

bool win32LoadEntireFile( const char* filename, void** data, uint32_t* numBytesRead )
{
    HANDLE file = CreateFileA( filename, GENERIC_READ, 0, 0, OPEN_EXISTING, 0, 0 );
    if ( ( file == INVALID_HANDLE_VALUE ) ) return false;

    DWORD fileSize = GetFileSize( file, 0 );
    if ( !fileSize ) return false;

    *data = HeapAlloc( GetProcessHeap(), 0, fileSize + 1 );
    if ( !*data ) return false;

    if ( !ReadFile( file, *data, fileSize, (LPDWORD)numBytesRead, 0 ) )
        return false;

    CloseHandle( file );
    ( (uint8_t*)*data )[ fileSize ] = 0;

    return true;
}

void Win32FreeFileData( void* data )
{
    HeapFree( GetProcessHeap(), 0, data );
}

#define SAFE_RELEASE(punk)  \
              if ((punk) != NULL)  \
                { (punk)->Release(); (punk) = NULL; }

class CMMNotificationClient : public IMMNotificationClient
{
    LONG _cRef;
    IMMDeviceEnumerator* _pEnumerator;

    // Private function to print device-friendly name
    HRESULT _PrintDeviceName( LPCWSTR  pwstrId );

public:
    CMMNotificationClient() :
        _cRef( 1 ),
        _pEnumerator( NULL )
    {
    }

    ~CMMNotificationClient()
    {
        SAFE_RELEASE( _pEnumerator )
    }

    // IUnknown methods -- AddRef, Release, and QueryInterface

    ULONG STDMETHODCALLTYPE AddRef()
    {
        return InterlockedIncrement( &_cRef );
    }

    ULONG STDMETHODCALLTYPE Release()
    {
        ULONG ulRef = InterlockedDecrement( &_cRef );
        if ( 0 == ulRef )
        {
            delete this;
        }
        return ulRef;
    }

    HRESULT STDMETHODCALLTYPE QueryInterface(
        REFIID riid, VOID** ppvInterface )
    {
        if ( IID_IUnknown == riid )
        {
            AddRef();
            *ppvInterface = (IUnknown*)this;
        }
        else if ( __uuidof( IMMNotificationClient ) == riid )
        {
            AddRef();
            *ppvInterface = (IMMNotificationClient*)this;
        }
        else
        {
            *ppvInterface = NULL;
            return E_NOINTERFACE;
        }
        return S_OK;
    }

    // Callback methods for device-event notifications.

    HRESULT STDMETHODCALLTYPE OnDefaultDeviceChanged(
        EDataFlow flow, ERole role,
        LPCWSTR pwstrDeviceId )
    {
        const char* pszFlow = "?????";
        const char* pszRole = "?????";

        _PrintDeviceName( pwstrDeviceId );

        switch ( flow )
        {
        case eRender:
            pszFlow = "eRender";
            break;
        case eCapture:
            pszFlow = "eCapture";
            break;
        }

        switch ( role )
        {
        case eConsole:
            pszRole = "eConsole";
            break;
        case eMultimedia:
            pszRole = "eMultimedia";
            break;
        case eCommunications:
            pszRole = "eCommunications";
            break;
        }

        printf( "  -->New default device: flow = %s, role = %s\n",
            pszFlow, pszRole );
        return S_OK;
    }

    HRESULT STDMETHODCALLTYPE OnDeviceAdded( LPCWSTR pwstrDeviceId )
    {
        _PrintDeviceName( pwstrDeviceId );

        printf( "  -->Added device\n" );
        return S_OK;
    };

    HRESULT STDMETHODCALLTYPE OnDeviceRemoved( LPCWSTR pwstrDeviceId )
    {
        _PrintDeviceName( pwstrDeviceId );

        printf( "  -->Removed device\n" );
        return S_OK;
    }

    HRESULT STDMETHODCALLTYPE OnDeviceStateChanged(
        LPCWSTR pwstrDeviceId,
        DWORD dwNewState )
    {
        const char* pszState = "?????";

        _PrintDeviceName( pwstrDeviceId );

        switch ( dwNewState )
        {
        case DEVICE_STATE_ACTIVE:
            pszState = "ACTIVE";
            break;
        case DEVICE_STATE_DISABLED:
            pszState = "DISABLED";
            break;
        case DEVICE_STATE_NOTPRESENT:
            pszState = "NOTPRESENT";
            break;
        case DEVICE_STATE_UNPLUGGED:
            pszState = "UNPLUGGED";
            break;
        }

        printf( "  -->New device state is DEVICE_STATE_%s (0x%8.8x)\n",
            pszState, dwNewState );

        return S_OK;
    }

    HRESULT STDMETHODCALLTYPE OnPropertyValueChanged(
        LPCWSTR pwstrDeviceId,
        const PROPERTYKEY key )
    {
        _PrintDeviceName( pwstrDeviceId );

        printf( "  -->Changed device property "
            "{%8.8x-%4.4x-%4.4x-%2.2x%2.2x-%2.2x%2.2x%2.2x%2.2x%2.2x%2.2x}#%d\n",
            key.fmtid.Data1, key.fmtid.Data2, key.fmtid.Data3,
            key.fmtid.Data4[ 0 ], key.fmtid.Data4[ 1 ],
            key.fmtid.Data4[ 2 ], key.fmtid.Data4[ 3 ],
            key.fmtid.Data4[ 4 ], key.fmtid.Data4[ 5 ],
            key.fmtid.Data4[ 6 ], key.fmtid.Data4[ 7 ],
            key.pid );
        return S_OK;
    }
};

#include "Functiondiscoverykeys_devpkey.h"

// Given an endpoint ID string, print the friendly device name.
HRESULT CMMNotificationClient::_PrintDeviceName( LPCWSTR pwstrId )
{
    HRESULT hr = S_OK;
    IMMDevice* pDevice = NULL;
    IPropertyStore* pProps = NULL;
    PROPVARIANT varString;

    CoInitialize( NULL );
    PropVariantInit( &varString );

    if ( _pEnumerator == NULL )
    {
        // Get enumerator for audio endpoint devices.
        hr = CoCreateInstance( __uuidof( MMDeviceEnumerator ),
            NULL, CLSCTX_INPROC_SERVER,
            __uuidof( IMMDeviceEnumerator ),
            (void**)&_pEnumerator );
    }
    if ( hr == S_OK )
    {
        hr = _pEnumerator->GetDevice( pwstrId, &pDevice );
    }
    if ( hr == S_OK )
    {
        hr = pDevice->OpenPropertyStore( STGM_READ, &pProps );
    }
    if ( hr == S_OK )
    {
        // Get the endpoint device's friendly-name property.
        hr = pProps->GetValue( PKEY_Device_FriendlyName, &varString );
    }
    printf( "----------------------\nDevice name: \"%S\"\n"
        "  Endpoint ID string: \"%S\"\n",
        ( hr == S_OK ) ? varString.pwszVal : L"null device",
        ( pwstrId != NULL ) ? pwstrId : L"null ID" );

    PropVariantClear( &varString );

    SAFE_RELEASE( pProps )
        SAFE_RELEASE( pDevice )
        CoUninitialize();
    return hr;
}

int mainAudioRender()
{
    void* fileBytes;
    uint32_t fileSize;
    bool result = win32LoadEntireFile( "Mat183.wav", &fileBytes, &fileSize );
    if ( !result ) {
        std::cerr << "win32LoadEntireFile fail" << std::endl;
    }
    assert( result );

    WavFile* wav = (WavFile*)fileBytes;
    // Check the Chunk IDs to make sure we loaded the file correctly
    assert( wav->riffId == 1179011410 );
    assert( wav->waveId == 1163280727 );
    assert( wav->fmtId == 544501094 );
    assert( wav->dataId == 1635017060 );
    // Check data is in format we expect
    assert( wav->formatCode == 1 ); // Only support PCM data
    assert( wav->numChannels == 2 ); // Only support 2-channel data
    assert( wav->fmtChunkSize == 16 ); // This should be true for PCM data
    assert( wav->sampleRate == 44100 ); // Only support 44100Hz data
    assert( wav->bitsPerSample == 16 ); // Only support 16-bit samples
    // This is how these fields are defined, no harm to assert that they're what we expect
    assert( wav->blockAlign == wav->numChannels * wav->bitsPerSample / 8 );
    assert( wav->byteRate == wav->sampleRate * wav->blockAlign );

    uint32_t numWavSamples = wav->dataChunkSize / ( wav->numChannels * sizeof( uint16_t ) );
    uint16_t* wavSamples = &wav->samples;


    const char* wavFilePath = "Mat183.wav";
    WAVEFORMATEX wfx = { 0 };
    BYTE* data = nullptr;
    DWORD dataSize = 0;

    if ( !LoadWAVFile( wavFilePath, wfx, data, dataSize ) ) {
        std::cerr << "Failed to load WAV file." << std::endl;
        return 1;
    }
    std::cout << "Loaded WAV file: " << wavFilePath << std::endl;

    std::cout << "wfx.wFormatTag: " << wfx.wFormatTag << std::endl;
    std::cout << "wfx.nChannels: " << wfx.nChannels << std::endl;
    std::cout << "wfx.nSamplesPerSec: " << wfx.nSamplesPerSec << std::endl;
    std::cout << "wfx.nAvgBytesPerSec: " << wfx.nAvgBytesPerSec << std::endl;
    std::cout << "wfx.nBlockAlign: " << wfx.nBlockAlign << std::endl;
    std::cout << "wfx.wBitsPerSample: " << wfx.wBitsPerSample << std::endl;
    std::cout << "wfx.cbSize: " << wfx.cbSize << std::endl;


    HRESULT hr = CoInitializeEx( nullptr, COINIT_SPEED_OVER_MEMORY );
    if ( FAILED(hr) ) {
        std::cerr << "CoInitializeEx failed:" << hr << std::endl;
    }
    assert( hr == S_OK );

    IMMDeviceEnumerator* deviceEnumerator;
    hr = CoCreateInstance( __uuidof( MMDeviceEnumerator ), nullptr, CLSCTX_ALL, __uuidof( IMMDeviceEnumerator ), (LPVOID*)( &deviceEnumerator ) );
    if ( FAILED( hr ) ) {
        std::cerr << "CoCreateInstance failed:" << hr << std::endl;
    }
    assert( hr == S_OK );

    CMMNotificationClient* notificationClient = new CMMNotificationClient();
    hr = deviceEnumerator->RegisterEndpointNotificationCallback( notificationClient );
    if ( FAILED( hr ) )
    {
        std::cerr << "RegisterEndpointNotificationCallback failed:" << hr << std::endl;
    }
    assert( hr == S_OK );


    std::cout << "Notification client registered. Press Enter to exit." << std::endl;
    std::cin.get();

    IMMDevice* audioDevice;
    hr = deviceEnumerator->GetDefaultAudioEndpoint( eRender, eConsole, &audioDevice );
    if ( FAILED( hr ) ) {
        std::cerr << "GetDefaultAudioEndpoint failed:" << hr << std::endl;
    }
    assert( hr == S_OK );

    //deviceEnumerator->Release();

    IAudioClient2* audioClient;
    hr = audioDevice->Activate( __uuidof( IAudioClient2 ), CLSCTX_ALL, nullptr, (LPVOID*)( &audioClient ) );
    if ( FAILED( hr ) ) {
        std::cerr << "Activate failed:" << hr << std::endl;
    }
    assert( hr == S_OK );

    audioDevice->Release();

    // WAVEFORMATEX* defaultMixFormat = NULL;
    // hr = audioClient->GetMixFormat(&defaultMixFormat);
    // assert(hr == S_OK);
    //if ( FAILED( hr ) )
    //{
    //    printf( "Unable to get mix format on audio client: %x.\n", hr );
    //    return false;
    //}

    WAVEFORMATEX mixFormat = {};
    mixFormat.wFormatTag = WAVE_FORMAT_PCM;
    mixFormat.nChannels = 2;
    mixFormat.nSamplesPerSec = 44100;//defaultMixFormat->nSamplesPerSec;
    mixFormat.wBitsPerSample = 16;
    mixFormat.nBlockAlign = ( mixFormat.nChannels * mixFormat.wBitsPerSample ) / 8;
    mixFormat.nAvgBytesPerSec = mixFormat.nSamplesPerSec * mixFormat.nBlockAlign;

    std::cout << "mixFormat.nSamplesPerSec:" << mixFormat.nSamplesPerSec << std::endl;
    std::cout << "mixFormat.nChannels:" << mixFormat.nChannels << std::endl;
    std::cout << "mixFormat.wBitsPerSample:" << mixFormat.wBitsPerSample << std::endl;
    std::cout << "mixFormat.nAvgBytesPerSec:" << mixFormat.nAvgBytesPerSec << std::endl;
    std::cout << "mixFormat.nBlockAlign:" << mixFormat.nBlockAlign << std::endl;
    std::cout << "mixFormat.cbSize:" << mixFormat.cbSize << std::endl;

    const int64_t REFTIMES_PER_SEC = 10000000; // hundred nanoseconds
    REFERENCE_TIME requestedSoundBufferDuration = REFTIMES_PER_SEC * 10;
    DWORD initStreamFlags = ( AUDCLNT_STREAMFLAGS_RATEADJUST
        | AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM
        | AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY );
    hr = audioClient->Initialize( AUDCLNT_SHAREMODE_SHARED,
        initStreamFlags,
        requestedSoundBufferDuration,
        0, &mixFormat, nullptr );
    assert( hr == S_OK );

    IAudioRenderClient* audioRenderClient;
    hr = audioClient->GetService( __uuidof( IAudioRenderClient ), (LPVOID*)( &audioRenderClient ) );
    if ( FAILED( hr ) ) {
        std::cerr << "GetService failed:" << hr << std::endl;
    }
    assert( hr == S_OK );

    UINT32 bufferSizeInFrames;
    hr = audioClient->GetBufferSize( &bufferSizeInFrames );
    if ( FAILED( hr ) ) {
        std::cerr << "GetBufferSize failed:" << hr << std::endl;
    }
    assert( hr == S_OK );

    hr = audioClient->Start();
    if ( FAILED( hr ) ) {
        std::cerr << "Start failed:" << hr << std::endl;
    }
    assert( hr == S_OK );

    double playbackTime = 0.0;
    const float TONE_HZ = 440;
    const int16_t TONE_VOLUME = 3000;
    int wavPlaybackSample = 0;

    std::cout << "Playing";

    while ( true )
    {
        // Padding is how much valid data is queued up in the sound buffer
        // if there's enough padding then we could skip writing more data
        UINT32 bufferPadding;
        hr = audioClient->GetCurrentPadding( &bufferPadding );
        if ( FAILED( hr ) ) {
            std::cerr << "GetCurrentPadding failed:" << hr << std::endl;
        }
        assert( hr == S_OK );

        // How much of our sound buffer we want to fill on each update.
        // Needs to be enough so that the playback doesn't reach garbage data
        // but we get less latency the lower it is (e.g. how long does it take
        // between pressing jump and hearing the sound effect)
        // Try setting this to e.g. bufferSizeInFrames / 250 to hear what happens when
        // we're not writing enough data to stay ahead of playback!
        UINT32 soundBufferLatency = bufferSizeInFrames; // / 50;
        UINT32 numFramesToWrite = soundBufferLatency - bufferPadding;

        int16_t* buffer;
        hr = audioRenderClient->GetBuffer( numFramesToWrite, (BYTE**)( &buffer ) );
        if ( FAILED( hr ) ) {
            std::cerr << "GetBuffer failed:" << hr << std::endl;
        }
        assert( hr == S_OK );

        if ( numFramesToWrite > 0 )
        {
            std::cout << ".";
        }
        for ( UINT32 frameIndex = 0; frameIndex < numFramesToWrite; ++frameIndex )
        {
            *buffer++ = wavSamples[ wavPlaybackSample ]; // left
            *buffer++ = wavSamples[ wavPlaybackSample ]; // right

            ++wavPlaybackSample;
            wavPlaybackSample %= numWavSamples;
        }
        hr = audioRenderClient->ReleaseBuffer( numFramesToWrite, 0 );
        if ( FAILED( hr ) ) {
            std::cerr << "ReleaseBuffer failed:" << hr << std::endl;
        }
        assert( hr == S_OK );

        // Get playback cursor position
        // This is good for visualising playback and seeing the reading/writing in action!
        IAudioClock* audioClock;
        hr = audioClient->GetService( __uuidof( IAudioClock ), (LPVOID*)( &audioClock ) );
        if ( FAILED( hr ) ) {
            std::cerr << "GetService failed:" << hr << std::endl;
        }
        UINT64 audioPlaybackFreq;
        UINT64 audioPlaybackPos;
        hr = audioClock->GetFrequency( &audioPlaybackFreq );
        if ( FAILED( hr ) ) {
            std::cerr << "GetFrequency failed:" << hr << std::endl;
        }
        //std::cout << "audioPlaybackFreq:" << audioPlaybackFreq << std::endl;

        hr = audioClock->GetPosition( &audioPlaybackPos, 0 );
        if ( FAILED( hr ) ) {
            std::cerr << "GetPosition failed:" << hr << std::endl;
        }
        audioClock->Release();
        UINT64 audioPlaybackPosInSeconds = audioPlaybackPos / audioPlaybackFreq;
        UINT64 audioPlaybackPosInSamples = audioPlaybackPosInSeconds * mixFormat.nSamplesPerSec;
    }

    audioClient->Stop();
    audioClient->Release();
    audioRenderClient->Release();

    Win32FreeFileData( fileBytes );

    return 0;
}

class CBase
{
public:
    std::wstring m_strName = L"";

    CBase()
    {
        printf( "CBase ctor\n" );
    };
    ~CBase()
    {
        printf( "CBase dtor\n" );
    };
};

class CNode : public CBase
{
public:
    CNode()
    {
        printf("CNode ctor\n");
    };
    ~CNode()
    {
        printf( "CNode dtor\n" );
    };
};


#include <iostream>
#include <filesystem>
#include <string>

//namespace fs = std::filesystem;
//
//std::string GetUniqueFilePath( const std::string& filePath ) {
//    fs::path path( filePath );
//    std::string directory = path.parent_path().string();
//    std::string fileNameWithoutExtension = path.stem().string();
//    std::string extension = path.extension().string();
//    std::string newFilePath = filePath;
//    int counter = 1;
//
//    while ( fs::exists( newFilePath ) ) {
//        newFilePath = directory + "/" + fileNameWithoutExtension + "(" + std::to_string( counter ) + ")" + extension;
//        counter++;
//    }
//
//    return newFilePath;
//}
#define _CRT_SECURE_NO_WARNINGS


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

#include <cmath>
#include <algorithm>

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


int Not__main()
{

    std::vector<int> vAA = { 1, 2, 3, 4, 5 };

    std::vector<int> vBB = std::vector<int> { vAA.begin(), vAA.end() };

    std::set< int > vSet{ vAA.begin(), vAA.end() };
    std::vector< int > vVec{ vSet.begin(), vSet.end() };
    for ( auto& i : vSet )
    {
        std::cout << i << std::endl;
    }

    for ( auto& i : vVec )
    { 
        std::cout << i << std::endl;
    }

    FILE* pFile = fopen( "testLittleEndian", "wb+" );
    if ( pFile )
    {
        uint32_t value = 0x12345678;
        fwrite( &value, sizeof( value ), 1, pFile );
        fclose( pFile );
    }

    pFile = fopen( "testLittleEndian", "r" );
    if ( pFile )
    {
        for ( int i = 0; i < sizeof( uint32_t ) / sizeof( uint8_t ); ++i )
        {
            uint8_t value;
            fread( &value, sizeof( value ), 1, pFile );
            printf( "%x ", value );
        }
        fclose( pFile );
    }

    //std::string filePath = "d:/dev/bakeNormal.png";
    //std::string uniqueFilePath = GetUniqueFilePath( filePath );
    //std::cout << uniqueFilePath << std::endl;


        CNode* pNode = new CNode;


    CoInitialize( nullptr );

    delete pNode;

    // Load your WAV file data
    BYTE* audioData = nullptr;
    DWORD dataSize = 0;
    WAVEFORMATEX waveFormat = {};

    LoadWAVFile( "Mat183.wav", waveFormat, audioData, dataSize );

    if ( audioData != nullptr ) {
        PlayWavFile( audioData, dataSize, waveFormat );
        delete[] audioData;
    }
    //getchar();

    /*
    return mainWaveout();
    /*/
    return mainAudioRender();
    //*/
}

bool LoadWAVFile( const char* filename, WAVEFORMATEX& wfx, BYTE*& data, DWORD& dataSize ) {
    std::ifstream file( filename, std::ios::binary );
    if ( !file ) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return false;
    }

    DWORD chunkId;
    DWORD chunkSize;
    DWORD format;
    DWORD subChunk1Id;
    DWORD subChunk1Size;
    DWORD subChunk2Id;
    DWORD subChunk2Size;

    file.read( reinterpret_cast<char*>( &chunkId ), sizeof( chunkId ) );
    file.read( reinterpret_cast<char*>( &chunkSize ), sizeof( chunkSize ) );
    file.read( reinterpret_cast<char*>( &format ), sizeof( format ) );
    file.read( reinterpret_cast<char*>( &subChunk1Id ), sizeof( subChunk1Id ) );
    file.read( reinterpret_cast<char*>( &subChunk1Size ), sizeof( subChunk1Size ) );

    if ( chunkId != 0x46464952 || format != 0x45564157 || subChunk1Id != 0x20746D66 ) { // "RIFF", "WAVE", "fmt "
        std::cerr << "Invalid WAV file format." << std::endl;
        return false;
    }

    file.read( reinterpret_cast<char*>( &wfx ), subChunk1Size );
    if ( subChunk1Size > sizeof( WAVEFORMATEX ) ) {
        file.seekg( subChunk1Size - sizeof( WAVEFORMATEX ), std::ios::cur ); // Skip extra bytes
    }

    file.read( reinterpret_cast<char*>( &subChunk2Id ), sizeof( subChunk2Id ) );
    file.read( reinterpret_cast<char*>( &subChunk2Size ), sizeof( subChunk2Size ) );

    if ( subChunk2Id != 0x61746164 ) { // "data"
        std::cerr << "Invalid WAV file data chunk." << std::endl;
        return false;
    }

    data = new BYTE[ subChunk2Size ];
    file.read( reinterpret_cast<char*>( data ), subChunk2Size );
    dataSize = subChunk2Size;

    file.close();
    return true;
}

void PlayWavFile( const BYTE* audioData, UINT32 dataSize, WAVEFORMATEX& waveFormat ) {
    HRESULT hr;
    IMMDeviceEnumerator* deviceEnumerator = nullptr;

    hr = CoCreateInstance( __uuidof( MMDeviceEnumerator ), nullptr, CLSCTX_ALL, __uuidof( IMMDeviceEnumerator ), (LPVOID*)( &deviceEnumerator ) );
    if ( FAILED( hr ) ) {
        std::cerr << "CoCreateInstance failed: " << hr << std::endl;
        return;
    }
    assert( hr == S_OK );

    CMMNotificationClient* notificationClient = new CMMNotificationClient();
    hr = deviceEnumerator->RegisterEndpointNotificationCallback( notificationClient );
    if ( FAILED( hr ) )
    {
        std::cerr << "RegisterEndpointNotificationCallback failed:" << hr << std::endl;
    }
    assert( hr == S_OK );


    std::cout << "Notification client registered. Press Enter to exit." << std::endl;
    std::cin.get();

    IMMDevice* audioDevice = nullptr;
    hr = deviceEnumerator->GetDefaultAudioEndpoint( eRender, eConsole, &audioDevice );
    if ( FAILED( hr ) ) {
        std::cerr << "GetDefaultAudioEndpoint failed: " << hr << std::endl;
        deviceEnumerator->Release();
        return;
    }
    assert( hr == S_OK );

    //deviceEnumerator->Release();

    IAudioClient* audioClient = nullptr;
    hr = audioDevice->Activate( __uuidof( IAudioClient ), CLSCTX_ALL, nullptr, (LPVOID*)( &audioClient ) );
    if ( FAILED( hr ) ) {
        std::cerr << "Activate failed: " << hr << std::endl;
        audioDevice->Release();
        return;
    }
    assert( hr == S_OK );

    //audioDevice->Release();

    // Initialize the audio client in shared mode
    //hr = audioClient->Initialize( AUDCLNT_SHAREMODE_SHARED, 0, 10000000, 0, &waveFormat, nullptr );
    //if ( FAILED( hr ) ) {
    //    std::cerr << "Initialize failed: " << hr << std::endl;
    //    audioClient->Release();
    //    return;
    //}
    //assert( hr == S_OK );

    const int bufferDurationSec = 2;
    const int64_t REFTIMES_PER_SEC = 10000000; // hundred nanoseconds
    REFERENCE_TIME requestedSoundBufferDuration = REFTIMES_PER_SEC * bufferDurationSec;
    DWORD initStreamFlags = ( AUDCLNT_STREAMFLAGS_RATEADJUST
        | AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM
        | AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY );
    hr = audioClient->Initialize( AUDCLNT_SHAREMODE_SHARED,
        initStreamFlags,
        requestedSoundBufferDuration,
        0, &waveFormat, nullptr );
    assert( hr == S_OK );

    // Get the buffer size
    UINT32 bufferFrameCount;
    hr = audioClient->GetBufferSize( &bufferFrameCount );
    if ( FAILED( hr ) ) {
        std::cerr << "GetBufferSize failed: " << hr << std::endl;
        audioClient->Release();
        return;
    }
    assert( hr == S_OK );

    // Get the render client
    IAudioRenderClient* renderClient = nullptr;
    hr = audioClient->GetService( __uuidof( IAudioRenderClient ), (LPVOID*)( &renderClient ) );
    if ( FAILED( hr ) ) {
        std::cerr << "GetService failed: " << hr << std::endl;
        audioClient->Release();
        return;
    }
    assert( hr == S_OK );

    // Start the audio stream
    hr = audioClient->Start();
    if ( FAILED( hr ) ) {
        std::cerr << "Start failed: " << hr << std::endl;
        renderClient->Release();
        audioClient->Release();
        return;
    }
    assert( hr == S_OK );

    // Fill the buffer with the audio data
    UINT32 numFramesAvailable;
    BYTE* pData;
    UINT32 numFramesPadding;
    UINT32 numFramesToWrite;
    const BYTE* currentPtr = audioData;
    UINT32 bytesRemaining = dataSize;

    while ( bytesRemaining > 0 ) {
        hr = audioClient->GetCurrentPadding( &numFramesPadding );
        if ( FAILED( hr ) ) {
            std::cerr << "GetCurrentPadding failed: " << hr << std::endl;
            break;
        }
        numFramesAvailable = bufferFrameCount - numFramesPadding;
        numFramesToWrite = std::min( bytesRemaining / waveFormat.nBlockAlign, numFramesAvailable );

        hr = renderClient->GetBuffer( numFramesToWrite, &pData );
        if ( FAILED( hr ) ) {
            std::cerr << "GetBuffer failed: " << hr << std::endl;
            break;
        }

        // Copy audio data to buffer
        memcpy( pData, currentPtr, numFramesToWrite * waveFormat.nBlockAlign );
        currentPtr += numFramesToWrite * waveFormat.nBlockAlign;
        bytesRemaining -= numFramesToWrite * waveFormat.nBlockAlign;

        hr = renderClient->ReleaseBuffer( numFramesToWrite, 0 );
        if ( FAILED( hr ) ) {
            std::cerr << "ReleaseBuffer failed: " << hr << std::endl;
            break;
        }

        Sleep( bufferDurationSec * 1000 ); // Add a small delay to avoid busy waiting
    }

    // Stop the audio stream
    hr = audioClient->Stop();
    if ( FAILED( hr ) ) {
        std::cerr << "Stop failed: " << hr << std::endl;
    }

    renderClient->Release();
    audioClient->Release();
}

void CleanupWAV( BYTE* data ) {
    delete[] data;
}


