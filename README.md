# 🔮 Keertana - Advanced Document Processing

Keertana is a powerful document processing system built with Google's latest Gemini 2.5 models and Files API. It provides advanced AI-powered document analysis, structured data extraction, and multimodal understanding capabilities.

## ✨ Features

### 🤖 Latest AI Models
- **Gemini 2.5 Pro Preview**: Most powerful thinking model with enhanced reasoning (1M token context)
- **Gemini 2.5 Flash Preview**: Fast performance model with excellent cost-efficiency
- **Gemini 2.0 Flash**: Next-generation features with cutting-edge capabilities

### 📄 Document Processing
- Upload files up to 2GB each (20GB total per project)
- Support for PDF, TXT, HTML, CSS, MD, CSV, XML, RTF, and more
- Native PDF vision processing with up to 1000 pages
- Automatic file cleanup after 48 hours

### 🧠 Analysis Capabilities
- **Natural Language Queries**: Ask questions about your documents
- **Structured Data Extraction**: Extract data according to custom schemas
- **Multi-Document Comparison**: Compare multiple documents simultaneously  
- **Batch Processing**: Process multiple documents concurrently
- **Strategic Insights**: Generate business insights and recommendations

## 🚀 Quick Start

### 1. Get Your Google API Key
1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Create a new API key
3. Set it as an environment variable:

```bash
# Linux/Mac
export GOOGLE_API_KEY='your-api-key-here'

# Windows
set GOOGLE_API_KEY=your-api-key-here
```

### 2. Install Dependencies
```bash
# The project uses astral-uv for dependency management
uv sync
```

### 3. Run the Demo
```bash
# Run the main demo with sample documents
uv run main.py

# Run comprehensive examples
uv run examples.py

# View configuration details
uv run config.py
```

## 📋 Usage Examples

### Basic Document Analysis
```python
from main import DocumentProcessor

# Initialize processor
processor = DocumentProcessor()

# Upload and analyze document
file = await processor.upload_document('document.pdf', 'My Document')
analysis = await processor.analyze_document(
    file, 
    'What are the key insights from this document?'
)
print(analysis)
```

### Structured Data Extraction
```python
# Define your data schema
schema = {
    "title": "string",
    "key_metrics": ["array of numbers"],
    "conclusions": ["array of strings"],
    "recommendations": "string"
}

# Extract structured data
data = await processor.extract_structured_data(file, schema, model='pro')
```

### Multi-Document Comparison
```python
# Upload multiple files
files = [
    await processor.upload_document('doc1.pdf'),
    await processor.upload_document('doc2.pdf'),
    await processor.upload_document('doc3.pdf')
]

# Compare documents
comparison = await processor.compare_documents(
    files,
    'What are the key differences between these documents?'
)
```

### Batch Processing
```python
# Process multiple documents with the same query
results = await processor.batch_analyze_documents(
    files,
    'Generate an executive summary of each document'
)
```

## 🏗️ Project Structure

```
keertana/
├── main.py           # Core DocumentProcessor class and demo
├── examples.py       # Comprehensive usage examples
├── config.py         # Model specifications and configuration
├── data/            # Sample documents (auto-generated)
├── pyproject.toml   # Project dependencies
└── README.md        # This file
```

## 📊 Model Comparison

| Model | Context | Speed | Cost | Best For |
|-------|---------|-------|------|----------|
| **Gemini 2.5 Pro** | 1M tokens | Slower | Higher | Complex analysis, strategic planning |
| **Gemini 2.5 Flash** | 1M tokens | Fast | Low | Rapid processing, batch operations |
| **Gemini 2.0 Flash** | 1M tokens | Fast | Variable | Innovation projects, multimodal tasks |

## 💰 Pricing (2025)

### Gemini 2.5 Pro Preview
- Input: $1.25-$2.50 per 1M tokens
- Output: $10.00-$15.00 per 1M tokens

### Gemini 2.5 Flash Preview  
- Input: $0.15 per 1M tokens
- Output: $0.60 per 1M tokens

### Files API
- **Free** - No cost for Files API usage
- Files auto-delete after 48 hours

## 🔧 Advanced Configuration

### Custom Model Selection
```python
# Use specific models for different tasks
processor = DocumentProcessor()

# Fast processing
result = await processor.analyze_document(file, query, model='flash')

# Complex reasoning
result = await processor.analyze_document(file, query, model='pro')

# Next-gen features
result = await processor.analyze_document(file, query, model='flash_2')
```

### Error Handling
```python
try:
    file = await processor.upload_document('large_file.pdf')
    analysis = await processor.analyze_document(file, 'Analyze this document')
except FileNotFoundError:
    print("File not found")
except ValueError as e:
    print(f"API key error: {e}")
except Exception as e:
    print(f"Processing error: {e}")
finally:
    processor.cleanup_files()  # Always cleanup
```

## 📚 Use Cases

### 📈 Financial Analysis
- Extract key financial metrics from reports
- Generate investment insights and risk assessments
- Compare financial performance across periods

### 🔬 Research Analysis
- Analyze academic papers and extract key findings
- Compare methodologies across research papers
- Generate literature reviews and summaries

### 🏢 Business Intelligence
- Process business reports and strategic documents
- Extract KPIs and performance metrics
- Generate executive summaries and recommendations

### 📋 Technical Documentation
- Analyze API documentation and extract endpoints
- Generate code examples and usage guides
- Compare different versions of documentation

## ⚡ Performance Tips

1. **Choose the Right Model**:
   - Use Flash for speed and cost-efficiency
   - Use Pro for complex reasoning tasks

2. **Optimize Prompts**:
   - Be specific in your instructions
   - Use structured output formats
   - Break complex tasks into steps

3. **Batch Processing**:
   - Process multiple documents simultaneously
   - Use async operations for better performance

4. **File Management**:
   - Files auto-delete after 48 hours
   - Manually cleanup when done to free resources

## 🛡️ Security Best Practices

- Store API keys in environment variables
- Never commit API keys to version control
- Validate file types before upload
- Monitor API usage and costs
- Implement proper error handling

## 🐛 Troubleshooting

### Common Issues

**API Key Error**:
```bash
export GOOGLE_API_KEY='your-actual-api-key'
```

**File Upload Error**:
- Check file size (max 2GB per file)
- Verify file format is supported
- Ensure file path is correct

**Model Not Available**:
- Check if model ID is correct
- Some models may have limited availability

**Rate Limiting**:
- Implement exponential backoff
- Monitor your API usage
- Consider upgrading your plan

## 📞 Support

For issues, questions, or contributions:
1. Check the configuration with `uv run config.py`
2. Review the examples in `examples.py`
3. Ensure your API key is properly set

## 🔮 Future Enhancements

- Support for more file formats
- Real-time document streaming
- Enhanced multimodal capabilities
- Custom model fine-tuning
- Advanced analytics dashboard

---

**Built with ❤️ using Google Gemini 2.5 and astral-uv**
