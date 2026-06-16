using System;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;
using UnityEngine.UI;

public class ZedImageUdpReceiver : MonoBehaviour
{
    public enum ImageDisplayMode
    {
        Mono,
        StereoSideBySide,
    }

    private const int HeaderSize = 16;
    private const byte Magic0 = (byte)'Z';
    private const byte Magic1 = (byte)'I';
    private const byte Magic2 = (byte)'M';
    private const byte Magic3 = (byte)'G';

    [Header("UDP Receiver")]
    [Tooltip("Python 端 --unity-image-port 对应的端口。")]
    public int receivePort = 5010;

    [Tooltip("多久没有收到完整帧后，在 Console 中提示一次。")]
    public float staleFrameTimeout = 1.0f;

    [Header("Display")]
    [Tooltip("可选：手动绑定单目显示 RawImage。留空且未绑定左右眼 RawImage 时，会自动在相机前创建一个世界空间画面板。")]
    public RawImage targetImage;

    [Tooltip("可选：手动绑定左眼 RawImage。StereoSideBySide 模式下显示输入图像左半边。")]
    public RawImage leftEyeImage;

    [Tooltip("可选：手动绑定右眼 RawImage。StereoSideBySide 模式下显示输入图像右半边。")]
    public RawImage rightEyeImage;

    [Tooltip("未绑定 RawImage 时自动创建显示面板。")]
    public bool autoCreateDisplay = true;

    [Tooltip("画面板跟随的相机；留空时使用 Camera.main。")]
    public Camera displayCamera;

    [Tooltip("自动创建的画面板距离相机多少米。")]
    public float displayDistance = 2.0f;

    [Tooltip("自动创建的画面板相对相机的垂直偏移，负数向下，单位米。")]
    public float displayYOffset = -0.25f;

    [Tooltip("自动创建的画面板俯仰角，降低画面后可调节到正对头显，单位度。")]
    public float displayPitchDegrees = 7.0f;

    [Tooltip("自动创建的画面板宽度，单位米。")]
    public float displayWidthMeters = 1.28f;

    [Tooltip("自动创建的画面板高度，单位米。")]
    public float displayHeightMeters = 0.96f;

    [Header("VR Stereo")]
    [Tooltip("Mono 显示整张图；StereoSideBySide 表示输入图像为左右眼横向拼接图。")]
    public ImageDisplayMode displayMode = ImageDisplayMode.Mono;

    [Tooltip("StereoSideBySide 使用的分眼采样 Shader。留空时自动查找 Custom/StereoSideBySideRawImage。")]
    public Shader stereoSideBySideShader;

    [Tooltip("左右眼显示反了时勾选，用于交换左右半图。")]
    public bool stereoSwapEyes = false;

    [Tooltip("是否打印接收帧状态。")]
    public bool logFrames = false;

    [Tooltip("是否在画面上显示 UDP 接收状态，便于 APK 调试。")]
    public bool showStatus = true;

    [Tooltip("可选：手动绑定用于显示接收状态的 Text。留空时自动创建。")]
    public Text statusText;

    [Tooltip("UDP 接收缓冲区大小，APK/WiFi 下建议保持较大。")]
    public int receiveBufferBytes = 4 * 1024 * 1024;

    [Tooltip("是否向 Python 图像发送端回传接收确认，便于判断 APK 是否收到 UDP。")]
    public bool sendAck = true;

    [Tooltip("ACK 回传间隔，单位秒。")]
    public float ackInterval = 1.0f;

    private static readonly DateTime UnixEpoch = new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc);

    private readonly object frameLock = new object();
    private readonly Dictionary<uint, FrameAssembly> frames = new Dictionary<uint, FrameAssembly>();
    private byte[] latestJpeg;
    private uint latestFrameId;
    private uint latestAppliedFrameId;
    private int packetCount;
    private int assembledFrameCount;
    private int decodedFrameCount;
    private int decodeFailCount;
    private double lastPacketTime;
    private double lastAckTime;
    private string lastRemote = "none";
    private string receiverError = "";

    private UdpClient udpClient;
    private Thread receiveThread;
    private volatile bool running;
    private Texture2D displayTexture;
    private Material stereoMaterial;
    private AspectRatioFitter aspectFitter;
    private double lastCompleteFrameTime;
    private float lastStatusLogTime;
    private bool stereoShaderWarningLogged;

    private class FrameAssembly
    {
        public readonly int ChunkCount;
        public readonly int TotalBytes;
        public readonly byte[][] Chunks;
        public readonly bool[] Seen;
        public int ReceivedChunks;
        public int ReceivedBytes;
        public double LastUpdateTime;

        public FrameAssembly(int chunkCount, int totalBytes, double now)
        {
            ChunkCount = chunkCount;
            TotalBytes = totalBytes;
            Chunks = new byte[chunkCount][];
            Seen = new bool[chunkCount];
            LastUpdateTime = now;
        }

        public bool IsComplete
        {
            get { return ReceivedChunks == ChunkCount && ReceivedBytes == TotalBytes; }
        }

        public byte[] Assemble()
        {
            byte[] result = new byte[TotalBytes];
            int offset = 0;
            for (int i = 0; i < Chunks.Length; i++)
            {
                byte[] chunk = Chunks[i];
                if (chunk == null)
                {
                    return null;
                }

                Buffer.BlockCopy(chunk, 0, result, offset, chunk.Length);
                offset += chunk.Length;
            }

            return result;
        }
    }

    private void Start()
    {
        Application.runInBackground = true;
        EnsureDisplayTarget();
        StartReceiver();
    }

    private void Update()
    {
        EnsureDisplayTarget();
        ApplyDisplayMaterial();
        ApplyLatestFrame();
        UpdateStatusText();
        LogStaleState();
    }

    private void StartReceiver()
    {
        if (running)
        {
            return;
        }

        try
        {
            udpClient = new UdpClient(AddressFamily.InterNetwork);
            udpClient.Client.SetSocketOption(SocketOptionLevel.Socket, SocketOptionName.ReuseAddress, true);
            udpClient.Client.ReceiveBufferSize = Mathf.Max(65536, receiveBufferBytes);
            udpClient.Client.Bind(new IPEndPoint(IPAddress.Any, receivePort));
            udpClient.Client.ReceiveTimeout = 100;
        }
        catch (Exception exc)
        {
            receiverError = exc.Message;
            Debug.LogError($"ZedImageUdpReceiver failed to listen on UDP :{receivePort}: {exc.Message}");
            return;
        }

        running = true;
        receiveThread = new Thread(ReceiveLoop)
        {
            IsBackground = true,
            Name = "ZedImageUdpReceiver",
        };
        receiveThread.Start();
        Debug.Log($"ZedImageUdpReceiver listening on UDP :{receivePort}");
    }

    private void ReceiveLoop()
    {
        IPEndPoint remote = new IPEndPoint(IPAddress.Any, 0);
        while (running)
        {
            try
            {
                byte[] packet = udpClient.Receive(ref remote);
                lock (frameLock)
                {
                    packetCount += 1;
                    lastPacketTime = NowSeconds();
                    lastRemote = remote.ToString();
                }
                HandlePacket(packet, remote);
            }
            catch (SocketException exc)
            {
                if (exc.SocketErrorCode != SocketError.TimedOut && running)
                {
                    Debug.LogWarning("ZedImageUdpReceiver socket error: " + exc.Message);
                }
            }
            catch (ObjectDisposedException)
            {
                break;
            }
            catch (Exception exc)
            {
                if (running)
                {
                    Debug.LogWarning("ZedImageUdpReceiver receive error: " + exc.Message);
                }
            }
        }
    }

    private void HandlePacket(byte[] packet, IPEndPoint remote)
    {
        if (packet == null || packet.Length <= HeaderSize || !HasMagic(packet))
        {
            return;
        }

        uint frameId = ReadUInt32LE(packet, 4);
        int chunkIndex = ReadUInt16LE(packet, 8);
        int chunkCount = ReadUInt16LE(packet, 10);
        int totalBytes = (int)ReadUInt32LE(packet, 12);
        int payloadBytes = packet.Length - HeaderSize;

        if (chunkCount <= 0 || chunkIndex < 0 || chunkIndex >= chunkCount || totalBytes <= 0 || payloadBytes <= 0)
        {
            return;
        }

        MaybeSendAck(remote, "packet", frameId, chunkIndex, chunkCount, totalBytes, NowSeconds());

        byte[] chunk = new byte[payloadBytes];
        Buffer.BlockCopy(packet, HeaderSize, chunk, 0, payloadBytes);
        double now = NowSeconds();

        lock (frameLock)
        {
            FrameAssembly frame;
            if (!frames.TryGetValue(frameId, out frame) || frame.ChunkCount != chunkCount || frame.TotalBytes != totalBytes)
            {
                frame = new FrameAssembly(chunkCount, totalBytes, now);
                frames[frameId] = frame;
            }

            frame.LastUpdateTime = now;
            if (!frame.Seen[chunkIndex])
            {
                frame.Seen[chunkIndex] = true;
                frame.Chunks[chunkIndex] = chunk;
                frame.ReceivedChunks += 1;
                frame.ReceivedBytes += payloadBytes;
            }

            if (frame.IsComplete)
            {
                byte[] jpeg = frame.Assemble();
                if (jpeg != null)
                {
                    latestJpeg = jpeg;
                    latestFrameId = frameId;
                    lastCompleteFrameTime = now;
                    assembledFrameCount += 1;
                    MaybeSendAck(remote, "complete", frameId, chunkIndex, chunkCount, totalBytes, now);
                }

                frames.Clear();
            }
            else
            {
                RemoveStaleFrames(now);
            }
        }
    }

    private void MaybeSendAck(IPEndPoint remote, string state, uint frameId, int chunkIndex, int chunkCount, int totalBytes, double now)
    {
        if (!sendAck || udpClient == null || remote == null)
        {
            return;
        }

        if (ackInterval > 0f && now - lastAckTime < ackInterval)
        {
            return;
        }

        try
        {
            string message = $"ZACK {state} frame={frameId} chunk={chunkIndex + 1}/{chunkCount} bytes={totalBytes} packets={packetCount} complete={assembledFrameCount} decoded={decodedFrameCount}";
            byte[] bytes = Encoding.ASCII.GetBytes(message);
            udpClient.Send(bytes, bytes.Length, remote);
            lastAckTime = now;
        }
        catch (Exception exc)
        {
            if (logFrames)
            {
                Debug.LogWarning("ZedImageUdpReceiver ACK send error: " + exc.Message);
            }
        }
    }

    private void ApplyLatestFrame()
    {
        byte[] jpeg = null;
        uint frameId = 0;
        lock (frameLock)
        {
            if (latestJpeg != null)
            {
                jpeg = latestJpeg;
                frameId = latestFrameId;
                latestJpeg = null;
            }
        }

        if (jpeg == null || !HasDisplayTarget())
        {
            return;
        }

        if (displayTexture == null)
        {
            displayTexture = new Texture2D(2, 2, TextureFormat.RGB24, false);
        }

        if (!displayTexture.LoadImage(jpeg, false))
        {
            lock (frameLock)
            {
                decodeFailCount += 1;
            }
            return;
        }

        ApplyTextureToDisplayTargets();
        lock (frameLock)
        {
            decodedFrameCount += 1;
            latestAppliedFrameId = frameId;
        }
        if (aspectFitter != null && displayTexture.height > 0)
        {
            float visibleWidth = displayMode == ImageDisplayMode.StereoSideBySide
                ? displayTexture.width * 0.5f
                : displayTexture.width;
            aspectFitter.aspectRatio = visibleWidth / displayTexture.height;
        }

        if (logFrames)
        {
            Debug.Log($"ZED frame {frameId}: {displayTexture.width}x{displayTexture.height}, {jpeg.Length} bytes");
        }
    }

    private bool HasDisplayTarget()
    {
        return targetImage != null || leftEyeImage != null || rightEyeImage != null;
    }

    private void ApplyTextureToDisplayTargets()
    {
        if (displayTexture == null)
        {
            return;
        }

        bool useSeparateEyeImages = displayMode == ImageDisplayMode.StereoSideBySide
            && (leftEyeImage != null || rightEyeImage != null);

        if (useSeparateEyeImages)
        {
            Rect leftRect = stereoSwapEyes
                ? new Rect(0.5f, 0f, 0.5f, 1f)
                : new Rect(0f, 0f, 0.5f, 1f);
            Rect rightRect = stereoSwapEyes
                ? new Rect(0f, 0f, 0.5f, 1f)
                : new Rect(0.5f, 0f, 0.5f, 1f);

            ApplyRawImageTexture(leftEyeImage, leftRect);
            ApplyRawImageTexture(rightEyeImage, rightRect);
            ApplyRawImageTexture(targetImage, new Rect(0f, 0f, 1f, 1f));
            return;
        }

        ApplyRawImageTexture(targetImage, new Rect(0f, 0f, 1f, 1f));
        ApplyRawImageTexture(leftEyeImage, new Rect(0f, 0f, 1f, 1f));
        ApplyRawImageTexture(rightEyeImage, new Rect(0f, 0f, 1f, 1f));
        ApplyDisplayMaterial();
    }

    private void ApplyRawImageTexture(RawImage image, Rect uvRect)
    {
        if (image == null)
        {
            return;
        }

        image.enabled = true;
        image.color = Color.white;
        image.texture = displayTexture;
        image.material = null;
        image.uvRect = uvRect;
    }

    private void EnsureDisplayTarget()
    {
        if (HasDisplayTarget() || !autoCreateDisplay)
        {
            return;
        }

        Camera cam = displayCamera != null ? displayCamera : Camera.main;
        GameObject canvasObject = new GameObject("ZedImageDisplayCanvas");
        RectTransform canvasRect = canvasObject.AddComponent<RectTransform>();
        Canvas canvas = canvasObject.AddComponent<Canvas>();
        canvasObject.AddComponent<CanvasScaler>();
        canvasObject.AddComponent<GraphicRaycaster>();

        float pixelWidth = Mathf.Max(1f, displayWidthMeters * 1000f);
        float pixelHeight = Mathf.Max(1f, displayHeightMeters * 1000f);
        canvasRect.sizeDelta = new Vector2(pixelWidth, pixelHeight);

        if (cam != null)
        {
            canvas.renderMode = RenderMode.WorldSpace;
            canvas.worldCamera = cam;
            canvasObject.transform.SetParent(cam.transform, false);
            canvasObject.transform.localPosition = new Vector3(0f, displayYOffset, displayDistance);
            canvasObject.transform.localRotation = Quaternion.Euler(displayPitchDegrees, 0f, 0f);
            canvasObject.transform.localScale = Vector3.one * 0.001f;
        }
        else
        {
            canvas.renderMode = RenderMode.ScreenSpaceOverlay;
        }

        GameObject imageObject = new GameObject("ZedImageRawImage");
        imageObject.transform.SetParent(canvasObject.transform, false);
        RectTransform imageRect = imageObject.AddComponent<RectTransform>();
        imageRect.anchorMin = Vector2.zero;
        imageRect.anchorMax = Vector2.one;
        imageRect.offsetMin = Vector2.zero;
        imageRect.offsetMax = Vector2.zero;

        targetImage = imageObject.AddComponent<RawImage>();
        targetImage.enabled = false;
        targetImage.color = Color.white;
        ApplyDisplayMaterial();
        aspectFitter = imageObject.AddComponent<AspectRatioFitter>();
        aspectFitter.aspectMode = AspectRatioFitter.AspectMode.FitInParent;
        aspectFitter.aspectRatio = Mathf.Max(0.01f, displayWidthMeters / Mathf.Max(0.01f, displayHeightMeters));

        if (showStatus)
        {
            EnsureStatusText(canvasObject.transform);
        }
    }

    private void EnsureStatusText(Transform parent)
    {
        if (statusText != null)
        {
            return;
        }

        GameObject textObject = new GameObject("ZedImageStatusText");
        textObject.transform.SetParent(parent, false);
        RectTransform textRect = textObject.AddComponent<RectTransform>();
        textRect.anchorMin = Vector2.zero;
        textRect.anchorMax = Vector2.one;
        textRect.offsetMin = new Vector2(24f, 18f);
        textRect.offsetMax = new Vector2(-24f, -18f);

        statusText = textObject.AddComponent<Text>();
        statusText.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
        statusText.fontSize = 28;
        statusText.alignment = TextAnchor.LowerLeft;
        statusText.color = new Color(0.9f, 0.95f, 1f, 1f);
        statusText.raycastTarget = false;
        statusText.text = $"Waiting for UDP image on :{receivePort}";

        Shadow shadow = textObject.AddComponent<Shadow>();
        shadow.effectColor = new Color(0f, 0f, 0f, 0.85f);
        shadow.effectDistance = new Vector2(2f, -2f);
        textObject.transform.SetAsLastSibling();
    }

    private void UpdateStatusText()
    {
        if (!showStatus || statusText == null)
        {
            return;
        }

        int packets;
        int assembled;
        int decoded;
        int failed;
        int pending;
        uint appliedFrame;
        double packetTime;
        double frameTime;
        string remote;
        string error;
        lock (frameLock)
        {
            packets = packetCount;
            assembled = assembledFrameCount;
            decoded = decodedFrameCount;
            failed = decodeFailCount;
            pending = frames.Count;
            appliedFrame = latestAppliedFrameId;
            packetTime = lastPacketTime;
            frameTime = lastCompleteFrameTime;
            remote = lastRemote;
            error = receiverError;
        }

        double now = NowSeconds();
        if (!string.IsNullOrEmpty(error))
        {
            statusText.text = $"UDP listen failed :{receivePort}\n{error}";
            statusText.color = new Color(1f, 0.45f, 0.35f, 1f);
            return;
        }

        if (decoded > 0 && now - frameTime < staleFrameTimeout)
        {
            statusText.text = $"UDP image OK  frame={appliedFrame}\nfrom {remote}";
            if (displayMode == ImageDisplayMode.StereoSideBySide)
            {
                statusText.text += "\nstereo: side-by-side";
            }
            statusText.color = new Color(0.45f, 1f, 0.55f, 0.9f);
            return;
        }

        if (packets <= 0)
        {
            statusText.text = $"Waiting for UDP image on :{receivePort}\nSet Python --unity-image-host to Quest IP";
            statusText.color = new Color(1f, 0.85f, 0.35f, 1f);
            return;
        }

        float sincePacket = packetTime > 0.0 ? (float)(now - packetTime) : -1f;
        statusText.text = $"UDP packets received, no fresh image\npackets={packets} complete={assembled} decoded={decoded} fail={failed} pending={pending}\nfrom {remote}, last={sincePacket:F1}s ago\nTry lower resolution / JPEG quality / Hz";
        statusText.color = new Color(1f, 0.85f, 0.35f, 1f);
    }

    private void RemoveStaleFrames(double now)
    {
        List<uint> staleIds = null;
        foreach (KeyValuePair<uint, FrameAssembly> item in frames)
        {
            if (now - item.Value.LastUpdateTime > 0.5)
            {
                if (staleIds == null)
                {
                    staleIds = new List<uint>();
                }
                staleIds.Add(item.Key);
            }
        }

        if (staleIds == null)
        {
            return;
        }

        for (int i = 0; i < staleIds.Count; i++)
        {
            frames.Remove(staleIds[i]);
        }
    }

    private void LogStaleState()
    {
        if (!logFrames || staleFrameTimeout <= 0f)
        {
            return;
        }

        double now = NowSeconds();
        if (lastCompleteFrameTime > 0.0 && now - lastCompleteFrameTime < staleFrameTimeout)
        {
            return;
        }

        if (Time.unscaledTime - lastStatusLogTime > staleFrameTimeout)
        {
            Debug.Log($"ZedImageUdpReceiver waiting for frames on UDP :{receivePort}");
            lastStatusLogTime = Time.unscaledTime;
        }
    }

    private static bool HasMagic(byte[] packet)
    {
        return packet.Length >= 4 && packet[0] == Magic0 && packet[1] == Magic1 && packet[2] == Magic2 && packet[3] == Magic3;
    }

    private static ushort ReadUInt16LE(byte[] packet, int offset)
    {
        return (ushort)(packet[offset] | (packet[offset + 1] << 8));
    }

    private static uint ReadUInt32LE(byte[] packet, int offset)
    {
        return (uint)packet[offset]
            | ((uint)packet[offset + 1] << 8)
            | ((uint)packet[offset + 2] << 16)
            | ((uint)packet[offset + 3] << 24);
    }

    private static double NowSeconds()
    {
        return DateTime.UtcNow.Subtract(UnixEpoch).TotalSeconds;
    }

    private void ApplyDisplayMaterial()
    {
        if (targetImage == null)
        {
            return;
        }

        if (displayMode == ImageDisplayMode.Mono || leftEyeImage != null || rightEyeImage != null)
        {
            targetImage.material = null;
            targetImage.uvRect = new Rect(0f, 0f, 1f, 1f);
            return;
        }

        Shader shader = stereoSideBySideShader != null
            ? stereoSideBySideShader
            : Shader.Find("Custom/StereoSideBySideRawImage");

        if (shader == null)
        {
            targetImage.material = null;
            if (!stereoShaderWarningLogged)
            {
                Debug.LogWarning("StereoSideBySide requires shader Custom/StereoSideBySideRawImage.");
                stereoShaderWarningLogged = true;
            }
            return;
        }

        if (stereoMaterial == null || stereoMaterial.shader != shader)
        {
            if (stereoMaterial != null)
            {
                Destroy(stereoMaterial);
            }
            stereoMaterial = new Material(shader)
            {
                name = "Stereo Side-by-Side RawImage Material",
            };
        }

        stereoMaterial.SetFloat("_SwapEyes", stereoSwapEyes ? 1f : 0f);
        targetImage.material = stereoMaterial;
    }

    private void OnDisable()
    {
        StopReceiver();
    }

    private void OnDestroy()
    {
        StopReceiver();
        if (stereoMaterial != null)
        {
            Destroy(stereoMaterial);
            stereoMaterial = null;
        }
    }

    private void OnApplicationQuit()
    {
        StopReceiver();
    }

    private void StopReceiver()
    {
        running = false;
        if (udpClient != null)
        {
            udpClient.Close();
            udpClient = null;
        }

        if (receiveThread != null && receiveThread.IsAlive)
        {
            receiveThread.Join(200);
        }
        receiveThread = null;
    }
}
