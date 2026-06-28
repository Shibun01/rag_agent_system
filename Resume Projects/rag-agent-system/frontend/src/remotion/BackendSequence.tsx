import { AbsoluteFill, interpolate, spring, useCurrentFrame, useVideoConfig } from 'remotion'

function MetricTile({ label, value, delay }: { label: string; value: string; delay: number }) {
  const frame = useCurrentFrame()
  const { fps } = useVideoConfig()
  const progress = spring({ frame: Math.max(0, frame - delay), fps, config: { damping: 14, stiffness: 110 } })

  return (
    <div
      style={{
        borderRadius: 24,
        padding: '18px 20px',
        background: 'rgba(255, 255, 255, 0.06)',
        border: '1px solid rgba(148, 163, 184, 0.18)',
        boxShadow: '0 20px 50px rgba(0, 0, 0, 0.25)',
        opacity: progress,
        transform: `translateY(${interpolate(progress, [0, 1], [18, 0])}px)`,
      }}
    >
      <div style={{ fontSize: 11, letterSpacing: '0.28em', textTransform: 'uppercase', color: '#94a3b8' }}>{label}</div>
      <div style={{ marginTop: 10, fontSize: 28, fontWeight: 700, color: '#f8fafc' }}>{value}</div>
    </div>
  )
}

export function BackendSequence() {
  const frame = useCurrentFrame()
  const { fps, width } = useVideoConfig()
  const titleProgress = spring({ frame, fps, config: { damping: 16, stiffness: 120 } })
  const glow = interpolate(frame, [0, 90, 180], [0.45, 0.8, 0.5])

  return (
    <AbsoluteFill
      style={{
        background:
          'radial-gradient(circle at 20% 20%, rgba(34, 211, 238, 0.24), transparent 28%), radial-gradient(circle at 80% 0%, rgba(16, 185, 129, 0.18), transparent 24%), linear-gradient(180deg, #06101d 0%, #091427 55%, #050b14 100%)',
        color: '#e2e8f0',
      }}
    >
      <div
        style={{
          position: 'absolute',
          inset: 28,
          borderRadius: 36,
          border: '1px solid rgba(148, 163, 184, 0.14)',
          background: `rgba(7, 16, 31, ${glow})`,
        }}
      />

      <div style={{ position: 'absolute', left: 48, top: 44, opacity: titleProgress, transform: `translateY(${interpolate(titleProgress, [0, 1], [18, 0])}px)` }}>
        <div style={{ fontSize: 12, letterSpacing: '0.35em', textTransform: 'uppercase', color: '#94a3b8' }}>RAG Agent System</div>
        <div style={{ marginTop: 12, fontSize: width > 900 ? 36 : 28, lineHeight: 1.08, fontWeight: 800, color: '#f8fafc' }}>Backend coverage with motion</div>
        <div style={{ marginTop: 12, maxWidth: 540, fontSize: 16, lineHeight: 1.65, color: '#cbd5e1' }}>
          Tailwind UI, RTK Query cache, and backend actions all moving together in one operator surface.
        </div>
      </div>

      <div style={{ position: 'absolute', left: 52, right: 52, top: 220, display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
        <MetricTile label="Health" value="Online" delay={8} />
        <MetricTile label="Collections" value="Live" delay={16} />
        <MetricTile label="Analytics" value="Tracked" delay={24} />
        <MetricTile label="Agents" value="Ready" delay={32} />
      </div>

      <div
        style={{
          position: 'absolute',
          left: 52,
          right: 52,
          bottom: 48,
          display: 'grid',
          gridTemplateColumns: 'repeat(4, 1fr)',
          gap: 16,
        }}
      >
        {[
          'Ingest',
          'RAG',
          'Chat',
          'Agents',
        ].map((label, index) => (
          <div
            key={label}
            style={{
              borderRadius: 22,
              padding: '16px 18px',
              background: 'rgba(255,255,255,0.05)',
              border: '1px solid rgba(148,163,184,0.14)',
              transform: `translateY(${interpolate(spring({ frame: Math.max(0, frame - index * 8), fps }), [0, 1], [14, 0])}px)`,
              opacity: spring({ frame: Math.max(0, frame - index * 8), fps }),
            }}
          >
            <div style={{ fontSize: 11, letterSpacing: '0.28em', textTransform: 'uppercase', color: '#94a3b8' }}>{label}</div>
            <div style={{ marginTop: 10, fontSize: 18, fontWeight: 700, color: '#f8fafc' }}>Backend endpoint</div>
          </div>
        ))}
      </div>

      <div
        style={{
          position: 'absolute',
          left: '50%',
          top: '52%',
          width: 92,
          height: 92,
          marginLeft: -46,
          marginTop: -46,
          borderRadius: 999,
          background: 'linear-gradient(135deg, #67e8f9 0%, #22c55e 100%)',
          boxShadow: '0 0 42px rgba(34, 211, 238, 0.5)',
          transform: `scale(${interpolate(spring({ frame, fps, config: { stiffness: 90, damping: 14 } }), [0, 1], [0.82, 1])})`,
        }}
      />
    </AbsoluteFill>
  )
}
