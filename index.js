import 'dotenv/config'
import express from 'express'
import cors from 'cors'
import multer from 'multer'
import path from 'path'
import fs from 'fs'
import { fileURLToPath } from 'url'
import ffmpegPath from 'ffmpeg-static'
import ffmpeg from 'fluent-ffmpeg'
import { v4 as uuidv4 } from 'uuid'
import OpenAI from 'openai'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

ffmpeg.setFfmpegPath(ffmpegPath)

const app = express()
app.use(cors())
app.use(express.json())

// ==================== RATE LIMITING ====================
const RATE_LIMIT_REQUESTS_PER_MINUTE = Number(process.env.RATE_LIMIT_PER_MINUTE || 20)
const RATE_LIMIT_WINDOW_MS = 60 * 1000 // 1 minuto

class RateLimiter {
  constructor(maxRequests, windowMs) {
    this.maxRequests = maxRequests
    this.windowMs = windowMs
    this.requests = []
  }

  canMakeRequest() {
    const now = Date.now()
    // Remove requisições antigas (fora da janela)
    this.requests = this.requests.filter((time) => now - time < this.windowMs)
    
    if (this.requests.length < this.maxRequests) {
      this.requests.push(now)
      return { allowed: true, waitTime: 0 }
    }
    
    // Calcula quanto tempo esperar até a próxima requisição
    const oldestRequest = this.requests[0]
    const waitTime = this.windowMs - (now - oldestRequest)
    return { allowed: false, waitTime: Math.ceil(waitTime / 1000) } // retorna em segundos
  }

  async waitIfNeeded() {
    const { allowed, waitTime } = this.canMakeRequest()
    if (!allowed) {
      await new Promise((resolve) => setTimeout(resolve, waitTime * 1000))
      return this.waitIfNeeded()
    }
  }
}

const rateLimiter = new RateLimiter(RATE_LIMIT_REQUESTS_PER_MINUTE, RATE_LIMIT_WINDOW_MS)

// ==================== RETRY COM BACKOFF ====================
const MAX_RETRIES = 3
const INITIAL_BACKOFF_MS = 1000 // 1 segundo
const MAX_BACKOFF_MS = 30000 // 30 segundos

async function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

async function retryWithBackoff(fn, retries = MAX_RETRIES) {
  let lastError
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      return await fn()
    } catch (error) {
      lastError = error
      
      // Se é erro de rate limit, espera mais tempo
      const isRateLimit = error.status === 429 || error.message?.includes('rate limit')
      
      if (attempt < retries) {
        // Backoff exponencial com jitter
        const baseDelay = Math.min(
          INITIAL_BACKOFF_MS * Math.pow(2, attempt),
          MAX_BACKOFF_MS
        )
        const jitter = Math.random() * 0.3 * baseDelay // até 30% de variação
        const delay = isRateLimit ? baseDelay * 2 : baseDelay + jitter
        
        console.log(`Tentativa ${attempt + 1} falhou. Tentando novamente em ${Math.round(delay)}ms...`)
        await sleep(delay)
      }
    }
  }
  throw lastError
}

// ==================== FILA DE PROCESSAMENTO ====================
class EvaluationQueue {
  constructor() {
    this.queue = []
    this.processing = false
    this.results = new Map() // jobId -> { status, result, error }
  }

  async add(jobId, taskFn) {
    return new Promise((resolve, reject) => {
      this.queue.push({ jobId, taskFn, resolve, reject })
      this.results.set(jobId, { status: 'pending' })
      this.processQueue()
    })
  }

  async processQueue() {
    if (this.processing || this.queue.length === 0) return
    
    this.processing = true
    
    while (this.queue.length > 0) {
      const { jobId, taskFn, resolve, reject } = this.queue.shift()
      
      try {
        // Aguarda rate limiter antes de processar
        await rateLimiter.waitIfNeeded()
        
        this.results.set(jobId, { status: 'processing' })
        console.log(`Processando avaliação: ${jobId}`)
        
        const result = await retryWithBackoff(taskFn)
        
        this.results.set(jobId, { status: 'completed', result })
        resolve(result)
      } catch (error) {
        console.error(`Erro ao processar ${jobId}:`, error)
        this.results.set(jobId, { status: 'failed', error: error.message })
        reject(error)
      }
    }
    
    this.processing = false
  }

  getStatus(jobId) {
    return this.results.get(jobId) || { status: 'not_found' }
  }
}

const evaluationQueue = new EvaluationQueue()

const uploadsDir = path.join(__dirname, 'uploads')
const framesDir = path.join(__dirname, 'frames')
fs.mkdirSync(uploadsDir, { recursive: true })
fs.mkdirSync(framesDir, { recursive: true })

app.use('/frames', express.static(framesDir))

const storage = multer.diskStorage({
  destination: (_req, _file, cb) => cb(null, uploadsDir),
  filename: (_req, file, cb) => cb(null, `${Date.now()}-${file.originalname}`),
})
const upload = multer({ storage })

app.post('/upload', upload.single('video'), async (req, res) => {
  try {
    const videoPath = req.file?.path
    const intervalSec = Number(req.body?.intervalSec || 1)
    if (!videoPath) return res.status(400).json({ error: 'Video ausente' })
    const id = uuidv4()
    const outDir = path.join(framesDir, id)
    fs.mkdirSync(outDir, { recursive: true })

    const fpsExpr = `fps=1/${Math.max(1, intervalSec)}`
    const outputPattern = path.join(outDir, 'frame-%04d.jpg')

    await new Promise((resolve, reject) => {
      ffmpeg(videoPath)
        .outputOptions(['-vf', fpsExpr, '-qscale:v', '2'])
        .output(outputPattern)
        .on('end', resolve)
        .on('error', reject)
        .run()
    })

    const files = fs
      .readdirSync(outDir)
      .filter((f) => /\.jpg$/i.test(f))
      .sort()
    const baseUrl = `${req.protocol}://${req.get('host')}`
    const frames = files.map((f) => `${baseUrl}/frames/${id}/${f}`)
    res.json({ id, frames })
  } catch (e) {
    res.status(500).json({ error: 'Falha ao extrair frames' })
  }
})

// Função que executa a avaliação (será chamada pela fila)
async function executeEvaluation(id, maxFrames) {
  const apiKey = process.env.OPENAI_API_KEY
  if (!apiKey) throw new Error('OPENAI_API_KEY ausente')

  const outDir = path.join(framesDir, id)
  if (!fs.existsSync(outDir)) throw new Error('frames não encontrados')

  const files = fs
    .readdirSync(outDir)
    .filter((f) => /\.jpg$/i.test(f))
    .sort()
    .slice(0, Math.max(1, Number(maxFrames)))

  const images = files.map((f) => {
    const p = path.join(outDir, f)
    const b64 = fs.readFileSync(p).toString('base64')
    return { filename: f, base64: b64 }
  })

  const client = new OpenAI({ apiKey })
  const content = [
    {
      type: 'text',
      text:
        'Você é um especialista em avaliação de veículos. Analise estas imagens de um veículo e avalie seu estado de conservação considerando os seguintes aspectos:\n\n' +
        '1. **Estado Geral**: Avalie a conservação geral do veículo (excelente, bom, regular, ruim)\n' +
        '2. **Carroceria**: Verifique amassados, riscos, arranhões, ferrugem, desalinhamento de painéis\n' +
        '3. **Pintura**: Avalie desbotamento, oxidação, repinturas mal feitas, diferenças de cor\n' +
        '4. **Vidros e Faróis**: Verifique rachaduras, trincas, desbotamento, quebras\n' +
        '5. **Pneus e Rodas**: Avalie desgaste, calibragem aparente, danos nas rodas\n' +
        '6. **Interior** (se visível): Estado dos bancos, painel, volante, limpeza geral\n' +
        '7. **Documentação/Placa**: Se visível, verifique se está legível e presente\n\n' +
        'Responda APENAS em JSON válido com o seguinte formato:\n' +
        '{\n' +
        '  "overall_score": 0-100,\n' +
        '  "conservation_status": "excelente" | "bom" | "regular" | "ruim",\n' +
        '  "bodywork_score": 0-100,\n' +
        '  "paint_score": 0-100,\n' +
        '  "glass_lights_score": 0-100,\n' +
        '  "tires_wheels_score": 0-100,\n' +
        '  "interior_score": 0-100 | null,\n' +
        '  "damages_detected": ["descrição do dano 1", "descrição do dano 2"],\n' +
        '  "legal_status": "aprovado" | "reprovado" | "condicionado",\n' +
        '  "legal_status_reason": "explicação do status legal",\n' +
        '  "recommendations": ["recomendação 1", "recomendação 2"],\n' +
        '  "best_frames": [{"filename": "string", "reason": "por que esta imagem é útil"}]\n' +
        '}\n\n' +
        'Critérios para status legal:\n' +
        '- **aprovado**: Veículo em bom estado, sem danos significativos que comprometam segurança ou legalidade\n' +
        '- **condicionado**: Veículo com problemas menores que podem ser corrigidos, mas não impedem uso\n' +
        '- **reprovado**: Veículo com danos graves, falta de documentação visível, ou problemas que comprometem segurança/legalidade',
    },
    ...images.map((img) => ({
      type: 'image_url',
      image_url: {
        url: `data:image/jpeg;base64,${img.base64}`,
      },
    })),
  ]

  const completion = await client.chat.completions.create({
    model: 'gpt-4o-mini',
    messages: [
      {
        role: 'system',
        content:
          'Você é um especialista em avaliação de veículos e inspeção veicular. Analise cuidadosamente as imagens fornecidas e forneça uma avaliação detalhada do estado de conservação do veículo. Responda APENAS em JSON válido, sem texto adicional antes ou depois.',
      },
      { role: 'user', content },
    ],
    temperature: 0.2,
  })

  const raw = completion.choices?.[0]?.message?.content || ''
  let parsed
  try {
    // Tenta parsear o JSON diretamente
    parsed = JSON.parse(raw)
    
    // Se o resultado parseado ainda contém um campo 'raw', tenta parsear novamente
    if (parsed.raw && typeof parsed.raw === 'string') {
      try {
        const nestedParsed = JSON.parse(parsed.raw)
        parsed = { ...parsed, ...nestedParsed }
        delete parsed.raw
      } catch (_) {
        // Mantém o raw se não conseguir parsear
      }
    }
  } catch (parseError) {
    // Se falhar, tenta extrair JSON de dentro de markdown code blocks
    const jsonMatch = raw.match(/```(?:json)?\s*([\s\S]*?)\s*```/) || raw.match(/\{[\s\S]*\}/)
    if (jsonMatch) {
      try {
        parsed = JSON.parse(jsonMatch[1] || jsonMatch[0])
      } catch (_) {
        parsed = { raw, error: 'Não foi possível parsear o JSON retornado' }
      }
    } else {
      parsed = { raw, error: 'Resposta não contém JSON válido' }
    }
  }
  
  return { id, files, result: parsed }
}

// Endpoint para adicionar avaliação na fila
app.post('/evaluate', async (req, res) => {
  try {
    const { id, maxFrames = 12 } = req.body || {}
    if (!id) return res.status(400).json({ error: 'id ausente' })

    // Adiciona na fila e retorna imediatamente
    evaluationQueue
      .add(id, () => executeEvaluation(id, maxFrames))
      .then((result) => {
        // Resultado será armazenado na fila e pode ser consultado via GET
        console.log(`Avaliação ${id} concluída com sucesso`)
      })
      .catch((error) => {
        console.error(`Avaliação ${id} falhou:`, error)
      })

    res.json({ 
      message: 'Avaliação adicionada à fila',
      jobId: id,
      status: 'pending'
    })
  } catch (e) {
    console.error('Erro ao adicionar avaliação na fila:', e)
    res.status(500).json({ error: 'Falha ao adicionar avaliação na fila' })
  }
})

// Endpoint para verificar status da avaliação (polling)
app.get('/evaluate/:id', async (req, res) => {
  try {
    const { id } = req.params
    const status = evaluationQueue.getStatus(id)
    
    if (status.status === 'not_found') {
      return res.status(404).json({ error: 'Avaliação não encontrada' })
    }
    
    if (status.status === 'completed') {
      return res.json({ 
        status: 'completed',
        result: status.result 
      })
    }
    
    if (status.status === 'failed') {
      return res.status(500).json({ 
        status: 'failed',
        error: status.error 
      })
    }
    
    // pending ou processing
    res.json({ 
      status: status.status,
      message: status.status === 'processing' ? 'Processando avaliação...' : 'Aguardando na fila...'
    })
  } catch (e) {
    console.error('Erro ao verificar status:', e)
    res.status(500).json({ error: 'Falha ao verificar status' })
  }
})

const port = Number(process.env.PORT || 5000)
app.listen(port, () => {})