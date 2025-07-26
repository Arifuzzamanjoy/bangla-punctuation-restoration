#!/usr/bin/env python3
"""
ULTRA-COMPREHENSIVE Bangla Data Collection from the ENTIRE INTERNET
This script demonstrates accessing ALL possible Bangla sources on the internet
"""

import sys
import os
import argparse
import logging
from datetime import datetime
import json

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.web_scraper import BanglaWebScraper
from config import GENERATION_CONFIG

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'ultra_comprehensive_scraping_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def show_all_available_sources():
    """Display ALL available Bangla sources that can be scraped"""
    
    scraper = BanglaWebScraper()
    
    print("🌐 ULTRA-COMPREHENSIVE BANGLA DATA SOURCES")
    print("=" * 80)
    print()
    
    total_sources = 0
    
    for category, sources in scraper.bangla_sources.items():
        print(f"📂 {category.upper().replace('_', ' ')} ({len(sources)} sources)")
        total_sources += len(sources)
        
        # Show first few sources as examples
        for i, source in enumerate(sources[:5]):
            print(f"   🔸 {source}")
        
        if len(sources) > 5:
            print(f"   ... and {len(sources) - 5} more sources")
        print()
    
    print(f"📡 RSS FEEDS ({len(scraper.rss_feeds)} feeds)")
    for i, feed in enumerate(scraper.rss_feeds[:5]):
        print(f"   🔸 {feed}")
    if len(scraper.rss_feeds) > 5:
        print(f"   ... and {len(scraper.rss_feeds) - 5} more RSS feeds")
    print()
    
    print(f"🔍 SEARCH PATTERNS ({len(scraper.search_patterns)} patterns)")
    for pattern in scraper.search_patterns:
        print(f"   🔸 {pattern}")
    print()
    
    total_sources += len(scraper.rss_feeds) + len(scraper.search_patterns)
    
    print("🎯 ADDITIONAL DATA SOURCES:")
    print("   📚 Internet Archive (Historical content)")
    print("   📱 Social Media Platforms (Reddit, Quora, etc.)")
    print("   🎓 Academic Repositories (ResearchGate, Google Scholar, etc.)")
    print("   🏛️ Government Documents and Portals")
    print("   🎥 Multimedia Transcripts (YouTube, streaming platforms)")
    print("   🔍 Search Engine Discovery (Google, Bing, DuckDuckGo)")
    print()
    
    print(f"📊 TOTAL ACCESSIBLE SOURCES: {total_sources}+ sources")
    print("   💡 Plus unlimited discovery through search engines!")
    print()
    
    print("✨ SCRAPING CAPABILITIES:")
    print("   🤖 Automatic Bangla text detection")
    print("   🧹 Quality filtering and deduplication")
    print("   ⏱️ Respectful rate limiting")
    print("   🔄 Error handling and retry mechanisms")
    print("   📊 Real-time progress tracking")
    print("   💾 Automatic data saving")
    print()

def demonstrate_maximum_scraping():
    """Demonstrate maximum internet scraping capabilities"""
    
    logger.info("🚀 ACTIVATING MAXIMUM INTERNET SCRAPING MODE")
    logger.info("=" * 80)
    
    scraper = BanglaWebScraper()
    
    # Show what we're about to scrape
    logger.info("📋 SOURCES TO BE SCRAPED:")
    
    source_count = 0
    for category, sources in scraper.bangla_sources.items():
        logger.info(f"   📂 {category}: {len(sources)} sources")
        source_count += len(sources)
    
    logger.info(f"   📡 RSS feeds: {len(scraper.rss_feeds)} feeds")
    logger.info(f"   🔍 Search patterns: {len(scraper.search_patterns)} patterns")
    
    total_potential = source_count + len(scraper.rss_feeds) + len(scraper.search_patterns)
    logger.info(f"   🎯 TOTAL POTENTIAL SOURCES: {total_potential}+")
    logger.info("")
    
    # Start comprehensive scraping
    logger.info("🌐 STARTING MAXIMUM INTERNET SCRAPING...")
    
    # Use the ultra-comprehensive method
    all_sentences = scraper.scrape_comprehensive_bangla_data(
        wikipedia_articles=300,  # More Wikipedia articles
        news_articles_per_site=30,  # More articles per site
        include_blogs=True,
        include_educational=True,
        include_all_internet=True  # THIS ACTIVATES MAXIMUM SCRAPING
    )
    
    logger.info(f"🎯 TOTAL SENTENCES COLLECTED: {len(all_sentences):,}")
    
    # Analysis
    if all_sentences:
        logger.info("📊 COLLECTION ANALYSIS:")
        logger.info(f"   📝 Average sentence length: {sum(len(s.split()) for s in all_sentences) / len(all_sentences):.1f} words")
        
        # Count punctuation types
        punct_count = {}
        for punct in ['।', '?', '!', ',', ';', ':']:
            count = sum(s.count(punct) for s in all_sentences)
            if count > 0:
                punct_count[punct] = count
        
        logger.info("   📊 Punctuation distribution:")
        for punct, count in punct_count.items():
            logger.info(f"      '{punct}': {count:,} occurrences")
        
        # Save comprehensive data
        output_data = {
            "collection_date": datetime.now().isoformat(),
            "total_sentences": len(all_sentences),
            "total_sources_accessed": total_potential,
            "scraping_method": "ULTRA_COMPREHENSIVE_INTERNET_WIDE",
            "sentence_analysis": {
                "avg_length": sum(len(s.split()) for s in all_sentences) / len(all_sentences),
                "punctuation_distribution": punct_count
            },
            "sample_sentences": all_sentences[:20],  # First 20 as samples
            "source_categories": list(scraper.bangla_sources.keys()),
            "capabilities": [
                "Major news portals (40+ sites)",
                "International Bangla news",
                "Literary websites and magazines", 
                "Educational and academic content",
                "Government documents",
                "Blog and forum content",
                "RSS feeds (30+ feeds)",
                "Social media content",
                "Internet Archive historical data",
                "Multimedia transcripts",
                "Search engine discovery"
            ]
        }
        
        with open('ultra_comprehensive_bangla_data.json', 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        logger.info("💾 Ultra-comprehensive data saved to 'ultra_comprehensive_bangla_data.json'")
        
        # Show some examples
        logger.info("📝 SAMPLE COLLECTED SENTENCES:")
        for i, sentence in enumerate(all_sentences[:5]):
            logger.info(f"   {i+1}. {sentence}")
    
    else:
        logger.warning("⚠️ No sentences collected - check internet connection and source availability")
    
    return all_sentences

def test_specific_source_category(category: str):
    """Test scraping from a specific source category"""
    
    scraper = BanglaWebScraper()
    
    if category not in scraper.bangla_sources:
        logger.error(f"❌ Category '{category}' not found!")
        logger.info(f"Available categories: {list(scraper.bangla_sources.keys())}")
        return
    
    sources = scraper.bangla_sources[category]
    logger.info(f"🎯 Testing {category} category ({len(sources)} sources)")
    
    all_sentences = []
    
    for i, source in enumerate(sources[:3]):  # Test first 3 sources
        logger.info(f"📂 Scraping {i+1}/{min(3, len(sources))}: {source}")
        
        try:
            sentences = scraper.scrape_url(source, "general")
            all_sentences.extend(sentences)
            logger.info(f"   ✅ Collected {len(sentences)} sentences")
            
        except Exception as e:
            logger.error(f"   ❌ Error: {e}")
    
    logger.info(f"🎯 Total from {category}: {len(all_sentences)} sentences")
    
    if all_sentences:
        logger.info("📝 Sample sentences:")
        for i, sentence in enumerate(all_sentences[:3]):
            logger.info(f"   {i+1}. {sentence}")

def main():
    parser = argparse.ArgumentParser(description='Ultra-comprehensive Bangla data collection from entire internet')
    parser.add_argument('--show-sources', action='store_true',
                       help='Show all available Bangla sources on the internet')
    parser.add_argument('--maximum-scraping', action='store_true',
                       help='Activate maximum internet scraping mode')
    parser.add_argument('--test-category', type=str,
                       help='Test scraping from specific category')
    parser.add_argument('--list-categories', action='store_true',
                       help='List all available source categories')
    
    args = parser.parse_args()
    
    if args.show_sources:
        show_all_available_sources()
    
    elif args.maximum_scraping:
        logger.info("⚠️ WARNING: Maximum scraping will access 100+ sources and may take hours!")
        response = input("Continue? (y/N): ")
        
        if response.lower() == 'y':
            sentences = demonstrate_maximum_scraping()
            logger.info(f"✅ Maximum scraping completed! Total: {len(sentences):,} sentences")
        else:
            logger.info("❌ Maximum scraping cancelled by user")
    
    elif args.test_category:
        test_specific_source_category(args.test_category)
    
    elif args.list_categories:
        scraper = BanglaWebScraper()
        logger.info("📂 Available source categories:")
        for category in scraper.bangla_sources.keys():
            logger.info(f"   🔸 {category}")
    
    else:
        parser.print_help()
        print("\n💡 USAGE EXAMPLES:")
        print("python scripts/ultra_comprehensive_scraping.py --show-sources")
        print("python scripts/ultra_comprehensive_scraping.py --maximum-scraping")
        print("python scripts/ultra_comprehensive_scraping.py --test-category major_news_portals")
        print("python scripts/ultra_comprehensive_scraping.py --list-categories")
        print()
        print("🌐 ACCESS TO ENTIRE INTERNET:")
        print("✅ 40+ Major Bangla news portals")
        print("✅ International Bangla news sources")
        print("✅ Educational and academic websites")
        print("✅ Government documents and portals")
        print("✅ Literary websites and magazines")
        print("✅ Blog and forum content")
        print("✅ Social media platforms")
        print("✅ RSS feeds (30+ feeds)")
        print("✅ Internet Archive historical data")
        print("✅ Multimedia transcripts")
        print("✅ Search engine discovery")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n⏹️ Scraping interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
