import { useState, useRef } from 'react'
import './App.css'

function App() {
  const [nailArtImage, setNailArtImage] = useState<string | null>(null)
  const [handImage, setHandImage] = useState<string | null>(null)
  const [resultImage, setResultImage] = useState<string | null>(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [error, setError] = useState<string | null>(null)
  
  const nailArtInputRef = useRef<HTMLInputElement>(null)
  const handInputRef = useRef<HTMLInputElement>(null)

  const handleImageUpload = (setter: (url: string | null) => void, e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      const url = URL.createObjectURL(file)
      setter(url)
      setError(null)
    }
  }

  const handleProcess = async () => {
    if (!nailArtImage || !handImage) {
      setError('请先上传美甲图片和手部照片')
      return
    }

    setIsProcessing(true)
    setError(null)
    setResultImage(null)

    try {
      // 调用后端 API
      const formData = new FormData()
      
      // 将 blob URL 转换为 file
      const nailArtBlob = await fetch(nailArtImage).then(r => r.blob())
      const handBlob = await fetch(handImage).then(r => r.blob())
      
      formData.append('nail_art', nailArtBlob, 'nail_art.jpg')
      formData.append('hand', handBlob, 'hand.jpg')

      const response = await fetch('/api/transfer', {
        method: 'POST',
        body: formData
      })

      if (!response.ok) {
        throw new Error('处理失败，请重试')
      }

      const data = await response.json()
      setResultImage(data.result_url)
    } catch (err) {
      setError(err instanceof Error ? err.message : '处理失败')
    } finally {
      setIsProcessing(false)
    }
  }

  const handleReset = () => {
    setNailArtImage(null)
    setHandImage(null)
    setResultImage(null)
    setError(null)
    if (nailArtInputRef.current) nailArtInputRef.current.value = ''
    if (handInputRef.current) handInputRef.current.value = ''
  }

  return (
    <div className="app">
      <header className="header">
        <h1>💅 美甲换手 AI</h1>
        <p className="subtitle">上传美甲款式 + 手部照片，AI 帮你换上美甲</p>
      </header>

      <main className="main">
        {!resultImage ? (
          <>
            <div className="upload-section">
              <div className="upload-box" onClick={() => nailArtInputRef.current?.click()}>
                {nailArtImage ? (
                  <img src={nailArtImage} alt="美甲图片" className="preview-image" />
                ) : (
                  <div className="upload-placeholder">
                    <span className="upload-icon">💅</span>
                    <span>点击上传美甲图片</span>
                    <span className="upload-hint">选择你想要的美甲款式</span>
                  </div>
                )}
                <input
                  ref={nailArtInputRef}
                  type="file"
                  accept="image/*"
                  onChange={(e) => handleImageUpload(setNailArtImage, e)}
                  hidden
                />
              </div>

              <div className="arrow">→</div>

              <div className="upload-box" onClick={() => handInputRef.current?.click()}>
                {handImage ? (
                  <img src={handImage} alt="手部照片" className="preview-image" />
                ) : (
                  <div className="upload-placeholder">
                    <span className="upload-icon">🖐️</span>
                    <span>点击上传手部照片</span>
                    <span className="upload-hint">拍摄或选择你的手部照片</span>
                  </div>
                )}
                <input
                  ref={handInputRef}
                  type="file"
                  accept="image/*"
                  onChange={(e) => handleImageUpload(setHandImage, e)}
                  hidden
                />
              </div>
            </div>

            {error && <div className="error-message">{error}</div>}

            <button 
              className="process-button"
              onClick={handleProcess}
              disabled={isProcessing || !nailArtImage || !handImage}
            >
              {isProcessing ? (
                <>
                  <span className="spinner"></span>
                  AI 正在处理中...
                </>
              ) : (
                '✨ 开始换美甲'
              )}
            </button>
          </>
        ) : (
          <div className="result-section">
            <h2>🎉 美甲换手完成！</h2>
            <div className="result-images">
              <div className="result-item">
                <img src={nailArtImage!} alt="原美甲" />
                <span>原美甲款式</span>
              </div>
              <div className="result-item">
                <img src={handImage!} alt="原手部" />
                <span>原手部照片</span>
              </div>
              <div className="result-item result-final">
                <img src={resultImage} alt="换甲效果" />
                <span>换甲效果 ✨</span>
              </div>
            </div>
            <div className="result-actions">
              <a href={resultImage} download="nail-art-result.jpg" className="download-button">
                📥 下载结果
              </a>
              <button onClick={handleReset} className="reset-button">
                🔄 再来一次
              </button>
            </div>
          </div>
        )}
      </main>

      <footer className="footer">
        <p>Powered by AI · Made with 💜</p>
      </footer>
    </div>
  )
}

export default App
