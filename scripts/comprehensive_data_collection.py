#!/usr/bin/env python3
"""
Comprehensive Bangla data collection script that demonstrates 
scraping from the entire internet for dataset generation
"""

import sys
import os
import argparse
import logging
from datetime import datetime
import json

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.dataset_generator import BanglaDatasetGenerator
from src.data.web_scraper import BanglaWebScraper
from src.data.data_processor import DataProcessor
from config import GENERATION_CONFIG, DATASET_CONFIG

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'comprehensive_data_collection_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def demonstrate_comprehensive_scraping():
    """Demonstrate comprehensive scraping capabilities"""
    
    logger.info("🌐 COMPREHENSIVE BANGLA DATA COLLECTION FROM THE INTERNET")
    logger.info("=" * 70)
    
    # Initialize the web scraper
    scraper = BanglaWebScraper()
    
    # Show what sources we can scrape from
    logger.info("📊 AVAILABLE DATA SOURCES:")
    logger.info("🔸 Wikipedia: 200+ articles from various categories")
    logger.info("🔸 News Portals: 20+ major Bangla news websites")
    logger.info("🔸 Magazines: Literary and general interest magazines")
    logger.info("🔸 Educational: Academic and reference websites")
    logger.info("🔸 Blogs & Forums: Community-generated content")
    logger.info("🔸 RSS Feeds: Real-time news content")
    logger.info("🔸 Literary Sites: Poetry, stories, novels")
    
    # Demonstrate scraping from different sources
    total_sentences = 0
    
    # 1. Wikipedia scraping
    logger.info("\n📚 SCRAPING WIKIPEDIA...")
    wiki_sentences = scraper.scrape_wikipedia_extensively(num_articles=50)  # Reduced for demo
    total_sentences += len(wiki_sentences)
    logger.info(f"✅ Collected {len(wiki_sentences)} sentences from Wikipedia")
    
    # 2. News portal scraping
    logger.info("\n📰 SCRAPING NEWS PORTALS...")
    news_sentences = scraper.scrape_news_portals_extensively(max_articles_per_site=5)  # Reduced for demo
    total_sentences += len(news_sentences)
    logger.info(f"✅ Collected {len(news_sentences)} sentences from news portals")
    
    # 3. RSS feeds
    logger.info("\n📡 SCRAPING RSS FEEDS...")
    rss_sentences = scraper.scrape_rss_feeds()
    total_sentences += len(rss_sentences)
    logger.info(f"✅ Collected {len(rss_sentences)} sentences from RSS feeds")
    
    # 4. Educational content
    logger.info("\n🎓 SCRAPING EDUCATIONAL CONTENT...")
    edu_sentences = scraper.scrape_educational_content()
    total_sentences += len(edu_sentences)
    logger.info(f"✅ Collected {len(edu_sentences)} sentences from educational sites")
    
    logger.info(f"\n🎯 TOTAL SENTENCES COLLECTED: {total_sentences}")
    
    # Save sample data
    all_sentences = wiki_sentences + news_sentences + rss_sentences + edu_sentences
    
    # Remove duplicates
    unique_sentences = list(set(all_sentences))
    logger.info(f"📊 Unique sentences after deduplication: {len(unique_sentences)}")
    
    # Save sample data
    sample_data = {
        "collection_date": datetime.now().isoformat(),
        "total_sources_scraped": 4,
        "total_sentences": len(unique_sentences),
        "sample_sentences": unique_sentences[:10],  # Save first 10 as sample
        "source_breakdown": {
            "wikipedia": len(wiki_sentences),
            "news_portals": len(news_sentences),
            "rss_feeds": len(rss_sentences),
            "educational": len(edu_sentences)
        }
    }
    
    with open('comprehensive_scraping_demo.json', 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)
    
    logger.info("💾 Sample data saved to 'comprehensive_scraping_demo.json'")
    
    return unique_sentences

def generate_comprehensive_dataset():
    """Generate a comprehensive dataset using all available sources"""
    
    logger.info("\n🚀 GENERATING COMPREHENSIVE DATASET")
    logger.info("=" * 50)
    
    # Initialize the enhanced dataset generator
    config = GENERATION_CONFIG.copy()
    config["min_sentences"] = 5000  # Reduced for demo
    config["wikipedia_articles"] = 100
    config["news_articles_per_site"] = 10
    
    generator = BanglaDatasetGenerator(config=config)
    
    # Generate the dataset
    logger.info("🔄 Starting comprehensive dataset generation...")
    dataset = generator.generate_dataset()
    
    if dataset is None:
        logger.error("❌ Failed to generate dataset")
        return None
    
    # Display statistics
    logger.info("\n📈 DATASET STATISTICS:")
    total_examples = 0
    for split_name, split_data in dataset.items():
        split_size = len(split_data)
        total_examples += split_size
        logger.info(f"  {split_name}: {split_size:,} examples")
    
    logger.info(f"  TOTAL: {total_examples:,} examples")
    
    # Quality validation
    logger.info("\n🔍 QUALITY VALIDATION:")
    processor = DataProcessor()
    quality_report = processor.validate_data_quality(dataset)
    
    for split_name, split_report in quality_report.items():
        logger.info(f"📊 {split_name} split quality:")
        logger.info(f"  ✅ Total examples: {split_report['total_examples']:,}")
        logger.info(f"  📏 Avg sentence length: {split_report['avg_sentence_length']:.1f} words")
        logger.info(f"  ⚠️  Quality issues: {split_report['num_quality_issues']}")
        
        # Show punctuation distribution
        punct_dist = split_report['punctuation_distribution']
        logger.info(f"  📝 Punctuation distribution:")
        for punct, count in punct_dist.items():
            if count > 0:
                logger.info(f"    '{punct}': {count:,}")
    
    # Save the dataset
    output_dir = "data/comprehensive_generated_dataset"
    logger.info(f"\n💾 Saving dataset to {output_dir}...")
    
    success = generator.save_dataset_locally(dataset, output_dir)
    if success:
        logger.info("✅ Dataset saved successfully!")
    else:
        logger.error("❌ Failed to save dataset")
        return None
    
    # Show some examples
    logger.info("\n📝 SAMPLE EXAMPLES:")
    train_data = dataset["train"]
    for i in range(min(3, len(train_data))):
        example = train_data[i]
        logger.info(f"Example {i+1}:")
        logger.info(f"  Unpunctuated: {example['unpunctuated_text']}")
        logger.info(f"  Punctuated:   {example['punctuated_text']}")
    
    return dataset

def main():
    parser = argparse.ArgumentParser(description='Comprehensive Bangla data collection demonstration')
    parser.add_argument('--demo-scraping', action='store_true',
                       help='Demonstrate comprehensive scraping capabilities')
    parser.add_argument('--generate-dataset', action='store_true',
                       help='Generate comprehensive dataset')
    parser.add_argument('--full-scale', action='store_true',
                       help='Run full-scale data collection (may take hours)')
    parser.add_argument('--output-dir', type=str, default='data/comprehensive_dataset',
                       help='Output directory for dataset')
    
    args = parser.parse_args()
    
    if args.demo_scraping:
        logger.info("🎯 DEMONSTRATION MODE: Comprehensive Internet Scraping")
        sentences = demonstrate_comprehensive_scraping()
        logger.info(f"✅ Demonstration completed! Collected {len(sentences)} sentences")
    
    elif args.generate_dataset:
        logger.info("🎯 DATASET GENERATION MODE")
        dataset = generate_comprehensive_dataset()
        if dataset:
            logger.info("✅ Dataset generation completed successfully!")
        else:
            logger.error("❌ Dataset generation failed!")
            return 1
    
    elif args.full_scale:
        logger.info("🎯 FULL-SCALE DATA COLLECTION MODE")
        logger.warning("⚠️  This will take several hours and collect massive amounts of data!")
        
        # Full-scale configuration
        config = GENERATION_CONFIG.copy()
        config["wikipedia_articles"] = 500
        config["news_articles_per_site"] = 50
        config["min_sentences"] = 100000  # 100K sentences
        
        generator = BanglaDatasetGenerator(config=config)
        dataset = generator.generate_dataset()
        
        if dataset:
            generator.save_dataset_locally(dataset, args.output_dir)
            logger.info(f"✅ Full-scale dataset saved to {args.output_dir}")
        else:
            logger.error("❌ Full-scale dataset generation failed!")
            return 1
    
    else:
        parser.print_help()
        logger.info("\n💡 USAGE EXAMPLES:")
        logger.info("python scripts/comprehensive_data_collection.py --demo-scraping")
        logger.info("python scripts/comprehensive_data_collection.py --generate-dataset")
        logger.info("python scripts/comprehensive_data_collection.py --full-scale")
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("\n⏹️  Data collection interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
