from transformers import pipeline
import torch
import logging

logger = logging.getLogger(__name__)

class FilingsSummarizer:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            
            # Auto-detect GPU availability
            device = 0 if torch.cuda.is_available() else -1
            device_name = f"GPU ({torch.cuda.get_device_name(0)})" if device == 0 else "CPU"
            
            cls._instance.summarizer = pipeline(
                "summarization",
                model="sshleifer/distilbart-cnn-12-6",
                device=device
            )
            logger.info(f"✅ DistilBART summarizer loaded on {device_name}")
        return cls._instance
    
    def summarize(self, text: str) -> str:
        if not text or len(text.strip()) < 50:  # Lower threshold
            logger.warning(f"Text too short: {len(text)} chars")
            return ""
        
        try:
            # Clean text
            text = text.strip()
            words = text.split()
            
            logger.info(f"Processing {len(words)} words")
            
            # Need at least 20 words
            if len(words) < 20:
                logger.warning(f"Too few words: {len(words)}")
                return ""
            
            # For short content, don't chunk - just summarize directly
            if len(words) <= 500:
                result = self.summarizer(
                    text,
                    max_length=min(len(words), 100),  # Don't exceed input length
                    min_length=min(20, len(words) // 2),
                    do_sample=False,
                    truncation=True
                )
                
                if result and len(result) > 0 and 'summary_text' in result[0]:
                    summary = result[0]['summary_text']
                    logger.info(f"Generated summary: {summary[:100]}...")
                    return f"1. {summary}"
                else:
                    logger.warning("No summary generated")
                    return ""
            
            # For longer content, use chunks
            chunks = [" ".join(words[i:i+1024]) for i in range(0, len(words), 1024)]
            
            summaries = []
            for idx, chunk in enumerate(chunks[:3]):
                if len(chunk.strip()) < 50:
                    continue
                
                logger.info(f"Summarizing chunk {idx+1}/{min(len(chunks), 3)}")
                    
                result = self.summarizer(
                    chunk, 
                    max_length=100, 
                    min_length=20,  # Lower minimum
                    do_sample=False,
                    truncation=True
                )
                
                if result and len(result) > 0 and 'summary_text' in result[0]:
                    summaries.append(result[0]['summary_text'])
            
            if not summaries:
                logger.warning("No summaries generated from chunks")
                return ""
            
            # Combine and format
            combined = ". ".join(summaries)
            sentences = [s.strip() for s in combined.split(".") if len(s.strip()) > 10]
            
            points = sentences[:20]
            return "\n".join([f"{i+1}. {s}." for i, s in enumerate(points)])
        
        except Exception as e:
            logger.error(f"Summary error: {e}", exc_info=True)
            return ""