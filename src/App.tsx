import { useState } from 'react'

type Tab = 'sensor' | 'gslab'

function App() {
  const [activeTab, setActiveTab] = useState<Tab>('sensor')

  return (
    <div style={{
      width: '100vw',
      height: '100vh',
      display: 'flex',
      flexDirection: 'column',
      backgroundColor: '#1a1a2e',
      color: '#e0e0e0',
      fontFamily: 'system-ui, -apple-system, sans-serif',
      overflow: 'hidden',
    }}>
      {/* Header */}
      <header style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '8px 16px',
        backgroundColor: '#16213e',
        borderBottom: '1px solid #0f3460',
        flexShrink: 0,
      }}>
        <h1 style={{ margin: 0, fontSize: '16px', fontWeight: 600 }}>
          Waymo Perception Studio
        </h1>
        <nav style={{ display: 'flex', gap: '4px' }}>
          <TabButton
            label="Sensor View"
            active={activeTab === 'sensor'}
            onClick={() => setActiveTab('sensor')}
          />
          <TabButton
            label="3DGS Lab 🧪"
            active={activeTab === 'gslab'}
            onClick={() => setActiveTab('gslab')}
          />
        </nav>
        <div style={{ fontSize: '12px', opacity: 0.6 }}>
          No segment loaded
        </div>
      </header>

      {/* Main Content */}
      <main style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        {activeTab === 'sensor' ? (
          <SensorView />
        ) : (
          <GSLabView />
        )}
      </main>

      {/* Timeline */}
      <footer style={{
        padding: '8px 16px',
        backgroundColor: '#16213e',
        borderTop: '1px solid #0f3460',
        flexShrink: 0,
      }}>
        <TimelinePlaceholder />
      </footer>
    </div>
  )
}

function TabButton({ label, active, onClick }: {
  label: string
  active: boolean
  onClick: () => void
}) {
  return (
    <button
      onClick={onClick}
      style={{
        padding: '4px 12px',
        fontSize: '13px',
        border: 'none',
        borderRadius: '4px',
        cursor: 'pointer',
        backgroundColor: active ? '#0f3460' : 'transparent',
        color: active ? '#e94560' : '#8892a0',
        fontWeight: active ? 600 : 400,
      }}
    >
      {label}
    </button>
  )
}

function SensorView() {
  return (
    <div style={{ flex: 1, display: 'grid', gridTemplateColumns: '1fr 1fr', gridTemplateRows: '1fr 1fr', gap: '2px', padding: '2px' }}>
      {/* Camera Panel */}
      <div style={{ backgroundColor: '#0f3460', borderRadius: '4px', display: 'flex', alignItems: 'center', justifyContent: 'center', opacity: 0.5 }}>
        Camera Views (5)
      </div>
      {/* BEV Panel */}
      <div style={{ backgroundColor: '#0f3460', borderRadius: '4px', display: 'flex', alignItems: 'center', justifyContent: 'center', opacity: 0.5 }}>
        Bird's Eye View
      </div>
      {/* LiDAR 3D View - spans full width */}
      <div style={{ backgroundColor: '#0f3460', borderRadius: '4px', display: 'flex', alignItems: 'center', justifyContent: 'center', gridColumn: '1 / -1', opacity: 0.5 }}>
        3D LiDAR View
      </div>
    </div>
  )
}

function GSLabView() {
  return (
    <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', opacity: 0.5 }}>
      3DGS Lab — Coming in Phase 4
    </div>
  )
}

function TimelinePlaceholder() {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '12px', fontSize: '13px' }}>
      <button style={{ background: 'none', border: 'none', color: '#e0e0e0', cursor: 'pointer', fontSize: '16px' }}>◀</button>
      <button style={{ background: 'none', border: 'none', color: '#e0e0e0', cursor: 'pointer', fontSize: '16px' }}>▶</button>
      <div style={{
        flex: 1,
        height: '4px',
        backgroundColor: '#0f3460',
        borderRadius: '2px',
        position: 'relative',
      }}>
        <div style={{
          width: '0%',
          height: '100%',
          backgroundColor: '#e94560',
          borderRadius: '2px',
        }} />
      </div>
      <span style={{ opacity: 0.6, fontSize: '12px' }}>0 / 0 frames</span>
      <span style={{ opacity: 0.6, fontSize: '12px' }}>1x</span>
    </div>
  )
}

export default App
