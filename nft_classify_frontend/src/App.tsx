import { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import axios from 'axios'
import { FiUpload, FiSearch, FiImage, FiArrowRight, FiGrid, FiInfo, FiCopy, FiExternalLink } from 'react-icons/fi'
import './App.css'

// Define types for our NFT results
interface NFTResult {
  score: number
  object_id: string
  collection_id: string
  name: string
  image_url: string
  nft_type?: string
  nft_collection_name?: string
  creator?: string
  description?: string
  created_time?: string
}

interface SearchResponse {
  results: NFTResult[]
  count: number
  query_time_ms: number
}

function App() {
  const [imageUrl, setImageUrl] = useState<string>('')
  const [urlInput, setUrlInput] = useState<string>('')
  const [isLoading, setIsLoading] = useState<boolean>(false)
  const [results, setResults] = useState<NFTResult[]>([])
  const [error, setError] = useState<string | null>(null)
  const [queryTime, setQueryTime] = useState<number | null>(null)

  // Handle file drop
  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      const file = acceptedFiles[0]
      const reader = new FileReader()
      
      reader.onload = () => {
        const dataUrl = reader.result as string
        setImageUrl(dataUrl)
        setUrlInput('')
      }
      
      reader.readAsDataURL(file)
    }
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.gif', '.webp', '.avif']
    },
    maxFiles: 1
  })

  // Handle URL input
  const handleUrlChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setUrlInput(e.target.value)
  }

  const handleUrlSubmit = () => {
    if (urlInput.trim()) {
      setImageUrl(urlInput)
    }
  }

  // Handle Enter key in URL input
  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      handleUrlSubmit()
    }
  }

  // Search for similar NFTs
  const handleSearch = async () => {
    if (!imageUrl) {
      setError('Please provide an image to search')
      return
    }

    setIsLoading(true)
    setError(null)
    
    try {
      const response = await axios.post<SearchResponse>('http://localhost:8003/search', {
        img_url: imageUrl,
        limit: 12,
        threshold: 0.5
      })
      
      setResults(response.data.results)
      setQueryTime(response.data.query_time_ms)
    } catch (err) {
      console.error('Search error:', err)
      setError('Error searching for similar NFTs. Please try again.')
    } finally {
      setIsLoading(false)
    }
  }

  // Format similarity score as percentage
  const formatScore = (score: number): string => {
    return (score * 100).toFixed(2) + '%'
  }

  return (
    <div className="w-full h-full flex flex-col bg-gradient-to-br from-gray-50 to-gray-100">
      {/* Enhanced Header */}
      <header className="py-5 px-8 bg-black text-white border-b border-gray-800 shadow-md">
        <div className="max-w-full w-full mx-auto px-4 sm:px-6 lg:px-8 flex justify-between items-center">
          <div className="flex items-center space-x-2">
            <div className="w-10 h-10 rounded-lg bg-white/10 flex items-center justify-center">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M12 22C17.5228 22 22 17.5228 22 12C22 6.47715 17.5228 2 12 2C6.47715 2 2 6.47715 2 12C2 17.5228 6.47715 22 12 22Z" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                <path d="M7.5 12L10.5 15L16.5 9" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
            </div>
            <h1 className="text-3xl font-bold tracking-tight">
              NFT<span className="text-gray-400 font-light">Similarity</span>
            </h1>
          </div>
          <div className="flex items-center space-x-4">
            <div className="text-xs bg-white/10 px-3 py-1.5 rounded-full text-gray-300">
              API Status: Online
            </div>
            <div className="text-xs bg-white/10 px-3 py-1.5 rounded-full text-gray-300 flex items-center">
              <span className="w-2 h-2 bg-green-400 rounded-full mr-2"></span>
              Connected
            </div>
          </div>
        </div>
      </header>

      <main className="flex-grow px-8 py-10 w-full overflow-x-hidden">
        <div className="max-w-full w-full mx-auto px-4 sm:px-6 lg:px-8">
          <div className="mb-12 max-w-3xl">
            <h2 className="text-2xl font-medium mb-3">Find visually similar NFTs</h2>
            <p className="text-gray-600">
              Upload an image or provide a URL to discover NFTs with similar visual characteristics in our database.
            </p>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-5 gap-8 mb-16">
            {/* Left Column - Upload Section */}
            <div className="lg:col-span-3">
              <div className="glass-card rounded-xl p-8 shadow-sm">
                <div 
                  {...getRootProps()} 
                  className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all h-64 flex flex-col items-center justify-center ${
                    isDragActive 
                      ? 'border-black bg-black/5 scale-[0.99]' 
                      : 'border-gray-200 hover:border-gray-400 hover:bg-gray-50'
                  }`}
                >
                  <input {...getInputProps()} />
                  <FiUpload className="text-5xl mb-4 text-gray-400" />
                  <p className="text-lg mb-2 font-medium">Drop image here, or click to browse</p>
                  <p className="text-sm text-gray-500">Supports JPEG, PNG, GIF, WebP, and AVIF</p>
                </div>

                <div className="mt-8">
                  <p className="text-sm uppercase tracking-wider text-gray-500 font-medium mb-3">Or use an image URL</p>
                  <div className="flex">
                    <input
                      type="text"
                      value={urlInput}
                      onChange={handleUrlChange}
                      onKeyDown={handleKeyDown}
                      placeholder="https://example.com/image.jpg"
                      className="flex-1 px-4 py-3 bg-white border border-gray-200 rounded-l-lg focus:outline-none focus:ring-2 focus:ring-black focus:border-transparent"
                    />
                    <button 
                      onClick={handleUrlSubmit}
                      className="px-4 py-3 bg-black text-white rounded-r-lg hover:bg-gray-800 transition-colors"
                    >
                      <FiArrowRight className="text-xl" />
                    </button>
                  </div>
                </div>
              </div>
            </div>

            {/* Right Column - Preview Section */}
            <div className="lg:col-span-2">
              <div className="glass-card rounded-xl p-8 h-full flex flex-col shadow-sm">
                <p className="text-sm uppercase tracking-wider text-gray-500 font-medium mb-4">Preview</p>
                
                <div className="flex-grow flex items-center justify-center bg-white/50 rounded-lg overflow-hidden mb-6 border border-gray-100">
                  {imageUrl ? (
                    <img 
                      src={imageUrl} 
                      alt="Preview" 
                      className="max-w-full max-h-64 object-contain"
                    />
                  ) : (
                    <div className="text-center text-gray-400">
                      <FiImage className="mx-auto text-6xl mb-4" />
                      <p className="text-sm">No image selected</p>
                    </div>
                  )}
                </div>

                <button
                  onClick={handleSearch}
                  disabled={!imageUrl || isLoading}
                  className={`w-full py-3.5 bg-black text-white font-medium rounded-lg hover:bg-gray-800 transition-colors flex items-center justify-center ${
                    !imageUrl || isLoading ? 'opacity-50 cursor-not-allowed' : ''
                  }`}
                >
                  {isLoading ? (
                    <span className="flex items-center">
                      <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                      Searching...
                    </span>
                  ) : (
                    <>
                      <FiSearch className="mr-2" />
                      <span>Search Similar NFTs</span>
                    </>
                  )}
                </button>
              </div>
            </div>
          </div>

          {/* Results Section */}
          {error && (
            <div className="bg-red-50 border border-red-100 text-red-800 rounded-xl p-4 mb-8 flex items-center">
              <FiInfo className="text-xl mr-2 flex-shrink-0" />
              <p>{error}</p>
            </div>
          )}

          {results.length > 0 && (
            <div className="mb-12">
              <div className="flex items-center justify-between mb-8">
                <div>
                  <h2 className="text-2xl font-bold mb-1">Search Results</h2>
                  <p className="text-gray-500 flex items-center">
                    <FiGrid className="mr-2" />
                    {results.length} similar NFTs found
                  </p>
                </div>
                {queryTime && (
                  <div className="text-sm bg-black text-white px-3 py-1.5 rounded-full flex items-center">
                    <svg className="w-4 h-4 mr-1.5" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                      <path d="M12 6V12L16 14" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                      <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2"/>
                    </svg>
                    {(queryTime / 1000).toFixed(2)}s
                  </div>
                )}
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-6 gap-6">
                {results.map((nft, index) => (
                  <div 
                    key={`${nft.object_id}-${index}`} 
                    className="glass-card rounded-xl overflow-hidden hover:shadow-md transition-all hover:translate-y-[-2px]"
                  >
                    <div className="aspect-square bg-white/50 overflow-hidden relative group">
                      <img 
                        src={nft.image_url} 
                        alt={nft.name || 'NFT'} 
                        className="w-full h-full object-contain"
                      />
                      <div className="absolute top-3 right-3">
                        <span className="text-xs font-medium bg-black text-white px-2 py-1 rounded-full">
                          {formatScore(nft.score)}
                        </span>
                      </div>
                      <div className="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                        <a 
                          href={nft.image_url} 
                          target="_blank" 
                          rel="noopener noreferrer"
                          className="w-10 h-10 rounded-full bg-white/20 flex items-center justify-center text-white mx-1 hover:bg-white/40 transition-colors"
                        >
                          <FiExternalLink />
                        </a>
                      </div>
                    </div>
                    
                    <div className="p-4">
                      <h3 className="font-bold text-lg truncate mb-1">
                        {nft.name || 'Unnamed NFT'}
                      </h3>
                      
                      {nft.nft_collection_name && (
                        <p className="text-sm text-gray-600 mb-2">
                          {nft.nft_collection_name}
                        </p>
                      )}
                      
                      <div className="flex justify-between items-center mt-3 pt-3 border-t border-gray-100">
                        {nft.creator && (
                          <div className="flex items-center">
                            <div className="w-5 h-5 rounded-full bg-gray-100 mr-2"></div>
                            <p className="text-xs text-gray-500">
                              {nft.creator.substring(0, 6)}...
                            </p>
                          </div>
                        )}
                        
                        <div className="flex items-center text-xs text-gray-400 font-mono">
                          <p>{nft.object_id.substring(0, 8)}...</p>
                          <button 
                            className="ml-1.5 text-gray-400 hover:text-gray-600 transition-colors"
                            onClick={() => navigator.clipboard.writeText(nft.object_id)}
                            title="Copy ID"
                          >
                            <FiCopy size={12} />
                          </button>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </main>

      <footer className="py-6 px-8 border-t border-gray-200 bg-white">
        <div className="max-w-full w-full mx-auto px-4 sm:px-6 lg:px-8 flex flex-col md:flex-row justify-between items-center">
          <div className="flex items-center mb-4 md:mb-0">
            <div className="w-6 h-6 rounded-md bg-black flex items-center justify-center mr-2">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M12 22C17.5228 22 22 17.5228 22 12C22 6.47715 17.5228 2 12 2C6.47715 2 2 6.47715 2 12C2 17.5228 6.47715 22 12 22Z" stroke="black" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                <path d="M7.5 12L10.5 15L16.5 9" stroke="black" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
            </div>
            <p className="text-sm text-gray-500">NFT Classification System</p>
          </div>
          <div className="flex items-center space-x-6">
            <a href="#" className="text-sm text-gray-500 hover:text-black transition-colors">API Docs</a>
            <a href="#" className="text-sm text-gray-500 hover:text-black transition-colors">About</a>
            <p className="text-sm text-gray-400 font-mono">&copy; {new Date().getFullYear()}</p>
          </div>
        </div>
      </footer>
    </div>
  )
}

export default App
