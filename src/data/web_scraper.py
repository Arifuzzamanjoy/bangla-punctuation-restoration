#!/usr/bin/env python3
"""
Enhanced web scraper for collecting Bangla text from diverse internet sources
"""

import os
import re
import json
import time
import random
import requests
import feedparser
import wikipedia
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Optional, Set
from urllib.parse import urljoin, urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import logging

# Handle config import for different execution contexts
try:
    from config import GENERATION_CONFIG
except ImportError:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from config import GENERATION_CONFIG

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BanglaWebScraper:
    """
    Enhanced web scraper for collecting Bangla text from diverse internet sources
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the web scraper
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or GENERATION_CONFIG
        self.session = requests.Session()
        
        # User agents for rotation
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:89.0) Gecko/20100101 Firefox/89.0'
        ]
        
        # COMPREHENSIVE list of Bangla sources from ENTIRE INTERNET
        self.bangla_sources = {
            "major_news_portals": [
                # Bangladesh Major News
                "https://www.prothomalo.com/",
                "https://www.kalerkantho.com/",
                "https://www.jugantor.com/",
                "https://www.ittefaq.com.bd/",
                "https://www.samakal.com/",
                "https://www.manabzamin.com/",
                "https://www.dainikamadershomoy.com/",
                "https://www.deshrupantor.com/",
                "https://www.inqilab.com/",
                "https://www.dailynayadiganta.com/",
                "https://www.thefinancialexpress.com.bd/",
                "https://www.thedailystar.net/bangla/",
                "https://www.newagebd.net/",
                "https://www.observerbd.com/",
                "https://www.dhakatribune.com/bangladesh",
                "https://bangla.bdnews24.com/",
                "https://www.risingbd.com/",
                "https://www.jagonews24.com/",
                "https://www.banglanews24.com/",
                "https://www.somoynews.tv/",
                "https://www.daily-bangladesh.com/",
                "https://www.dailysangram.com/",
                "https://www.ajkerpatrika.com/",
                "https://www.natunbarta.com/",
                "https://www.amaderdeshonline.com/",
                "https://www.dailyjanakantha.com/",
                "https://www.dainikduniya.com/",
                "https://www.protidiner.com/",
                "https://www.shirshanews.com/",
                "https://www.ekattor.tv/",
                "https://www.channel24bd.tv/",
                "https://www.ntv.com.bd/",
                "https://www.rtvonline.com/",
                "https://www.jamuna.tv/",
                "https://www.somoy.tv/",
                "https://www.atnbangla.tv/",
                "https://www.independent.bd/",
                "https://www.mzamin.com/",
                "https://www.dailysun.com/",
                "https://www.dailyasianage.com/",
                "https://www.financialexpress.com.bd/",
                "https://www.dhakatimes24.com/",
                "https://www.ekushey-tv.com/",
                "https://www.dainikshiksha.com/"
            ],
            "international_bangla_news": [
                # International Bangla News
                "https://www.bbc.com/bengali",
                "https://bangla.voanews.com/",
                "https://www.anandabazar.com/",
                "https://www.eisamay.com/",
                "https://www.sangbadpratidin.in/",
                "https://www.bartamanbarta.com/",
                "https://www.khabar365din.com/",
                "https://www.thewall.in/bengali/",
                "https://bengali.news18.com/",
                "https://www.aajkaal.in/",
                "https://www.uttarbanga.com/",
                "https://www.dainikstatesman.com/",
                "https://www.ganashakti.com/",
                "https://www.millenniumpost.in/bengali",
                "https://www.dainikbhaskar.com/bengali/",
                "https://www.zeenews.india.com/bengali/",
                "https://bengali.asianetnews.com/",
                "https://www.republicworld.com/bengali/",
                "https://bn.wikipedia.org/",
                "https://www.dw.com/bn/",
                "https://www.dailymuslim.com.bd/"
            ],
            "magazines": [
                # Literary & General Magazines
                "https://www.anandalokmagazine.com/",
                "https://www.unmadmagazine.com/",
                "https://www.kishore-alo.com/",
                "https://www.shishu.com.bd/",
                "https://www.dhaka.prokash.com/",
                "https://www.shaptahikbichitra.com/",
                "https://www.rokomari.com/book/author/",
                "https://www.shandhani.com/",
                "https://www.sachitra.com/",
                "https://www.banglamagazine.com/",
                "https://www.shomoyer.com/",
                "https://www.chowdhurybazar.com/"
            ],
            "literary_sites": [
                # Literature & Poetry
                "https://www.banglabook.net/",
                "https://www.golpoguccho.com/",
                "https://www.kobita.com.bd/",
                "https://www.sahittyokarmi.com/",
                "https://www.prabasi.org/",
                "https://www.choturmatrik.com/",
                "https://www.arun.net/",
                "https://www.lekhok.org/",
                "https://www.kobitakotha.com/",
                "https://www.banglalyrics.com/",
                "https://www.bangla-book.com/",
                "https://www.bdebooks.com/",
                "https://www.onubad.com/",
                "https://www.sahityapatrika.com/",
                "https://www.kobiguru.com/",
                "https://www.golpopatrika.com/",
                "https://www.natokpala.com/",
                "https://www.banglanataka.com/"
            ],
            "educational": [
                # Educational & Academic
                "https://bn.wikipedia.org/",
                "https://www.banglapedia.org/",
                "https://www.rokomari.com/book",
                "https://www.boekhaat.com/",
                "https://www.10minuteschool.com/",
                "https://www.studypress.org/",
                "https://www.udvash.com/",
                "https://www.school.edu.bd/",
                "https://www.bangla.gov.bd/",
                "https://www.moedu.gov.bd/",
                "https://www.ugc.gov.bd/",
                "https://www.du.ac.bd/",
                "https://www.buet.ac.bd/",
                "https://www.cu.ac.bd/",
                "https://www.ru.ac.bd/",
                "https://www.nu.ac.bd/",
                "https://www.nctb.gov.bd/",
                "https://www.education.gov.bd/",
                "https://www.hseb.edu.bd/",
                "https://www.teachers.gov.bd/"
            ],
            "blogs_forums": [
                # Blogs & Forums
                "https://www.amarblog.com/",
                "https://www.somewhereinblog.net/",
                "https://www.istishon.com/",
                "https://medium.com/tag/bangla",
                "https://www.sachalayatan.com/",
                "https://www.chintadhara.org/",
                "https://www.ektushobdo.com/",
                "https://www.banglatribune.com/",
                "https://www.banglacricket.com/",
                "https://www.charbak.com/",
                "https://www.shorol.com/",
                "https://www.nirbachito.com/",
                "https://www.techshohor.com/",
                "https://www.priyojon.com/",
                "https://www.olpokotha.com/"
            ],
            "government_portals": [
                # Government & Official
                "https://www.bangladesh.gov.bd/",
                "https://www.mof.gov.bd/",
                "https://www.cabinet.gov.bd/",
                "https://www.mopa.gov.bd/",
                "https://www.pmo.gov.bd/",
                "https://www.parliament.gov.bd/",
                "https://www.mochta.gov.bd/",
                "https://www.mincom.gov.bd/",
                "https://www.ictd.gov.bd/",
                "https://www.bsti.gov.bd/"
            ],
            "business_finance": [
                # Business & Finance
                "https://www.bb.org.bd/",
                "https://www.bsec.gov.bd/",
                "https://www.dse.com.bd/",
                "https://www.cse.com.bd/",
                "https://www.businessnews24.com/",
                "https://www.bonikbarta.net/",
                "https://www.sharebazar.com/",
                "https://www.economicobserver.com.bd/"
            ],
            "tech_science": [
                # Technology & Science
                "https://www.techshohor.com/",
                "https://www.techtunes.io/",
                "https://www.tech.com.bd/",
                "https://www.ictworld.com/",
                "https://www.digitalbangladesh.gov.bd/",
                "https://www.basis.org.bd/",
                "https://www.bcs.org.bd/"
            ],
            "entertainment": [
                # Entertainment & Culture
                "https://www.banglacinema.com/",
                "https://www.chorki.com/",
                "https://www.hoichoi.tv/",
                "https://www.bongodorshon.com/",
                "https://www.banglamusic.com/",
                "https://www.banglasong.com/",
                "https://www.natokghar.com/"
            ],
            "sports": [
                # Sports
                "https://www.banglacricket.com/",
                "https://www.tigercricket.com.bd/",
                "https://www.footballbd.com/",
                "https://www.khelajog.com/"
            ],
            "religion_culture": [
                # Religion & Culture
                "https://www.islamhouse.com/bn/",
                "https://www.peacetv.tv/",
                "https://www.al-islam.org/bn/",
                "https://www.askislampedia.com/bn/",
                "https://www.hadithbd.com/",
                "https://www.quranmajid.com/",
                "https://www.islamicfinder.org/bn/"
            ]
        }
        
        # COMPREHENSIVE RSS feeds from ALL major Bangla sources
        self.rss_feeds = [
            # Major News RSS
            "https://www.prothomalo.com/feed/",
            "https://www.kalerkantho.com/feed/",
            "https://www.jugantor.com/feed/",
            "https://bangla.bdnews24.com/rss.xml",
            "https://www.jagonews24.com/rss.xml",
            "https://www.bbc.com/bengali/index.xml",
            "https://www.samakal.com/feed/",
            "https://www.ittefaq.com.bd/feed/",
            "https://www.manabzamin.com/feed/",
            "https://www.risingbd.com/feed/",
            "https://www.banglanews24.com/rss.xml",
            "https://www.somoynews.tv/feed/",
            "https://www.daily-bangladesh.com/feed/",
            "https://www.natunbarta.com/feed/",
            "https://www.ekattor.tv/feed/",
            "https://www.jamuna.tv/feed/",
            "https://www.somoy.tv/feed/",
            "https://www.independent.bd/feed/",
            "https://www.mzamin.com/feed/",
            "https://www.dailysun.com/feed/",
            # International Bangla RSS
            "https://www.anandabazar.com/feed/",
            "https://www.eisamay.com/feed/",
            "https://www.sangbadpratidin.in/feed/",
            "https://bengali.news18.com/feed/",
            "https://www.aajkaal.in/feed/",
            "https://www.ganashakti.com/feed/",
            # Blog RSS
            "https://www.amarblog.com/feed/",
            "https://www.somewhereinblog.net/feed/",
            "https://www.sachalayatan.com/feed/",
            "https://www.techshohor.com/feed/",
            "https://medium.com/feed/tag/bangla"
        ]
        
        # Search engines and web crawling patterns for Bangla content
        self.search_patterns = [
            # Google search for Bangla content
            "https://www.google.com/search?q=site%3A.bd+intext%3A\"বাংলা\"&lr=lang_bn",
            "https://www.google.com/search?q=site%3A.com+intext%3A\"বাংলাদেশ\"&lr=lang_bn",
            "https://www.google.com/search?q=filetype%3Apdf+\"বাংলা\"&lr=lang_bn",
            # Bing search for Bangla
            "https://www.bing.com/search?q=site%3A.bd+language%3Abn",
            # Duck Duck Go
            "https://duckduckgo.com/?q=site%3A.bd+বাংলা&lr=web_region_wt_BD"
        ]
        
        # Content selectors for different types of sites
        self.content_selectors = {
            "news": [
                ".story-content", ".article-body", ".content", ".post-content",
                "article p", ".news-content", ".story-details", ".article-content",
                ".entry-content", ".post-body", ".article-text", ".news-body"
            ],
            "blog": [
                ".post-content", ".entry-content", ".article-content", 
                ".blog-content", ".content", "article p", ".post-body"
            ],
            "wiki": [
                ".mw-parser-output p", ".content p", "#mw-content-text p"
            ],
            "general": [
                "p", "article", ".content", ".article", ".post", ".story"
            ]
        }
    
    def get_random_headers(self) -> Dict[str, str]:
        """Get random headers for requests"""
        return {
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'bn-BD,bn;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
    
    def is_bangla_text(self, text: str, min_bangla_ratio: float = 0.6) -> bool:
        """Check if text contains significant Bangla content"""
        if not text or len(text.strip()) < 10:
            return False
        
        bangla_chars = sum(1 for c in text if '\u0980' <= c <= '\u09FF')
        total_chars = sum(1 for c in text if c.isalpha())
        
        if total_chars == 0:
            return False
        
        bangla_ratio = bangla_chars / total_chars
        return bangla_ratio >= min_bangla_ratio
    
    def extract_sentences_from_text(self, text: str) -> List[str]:
        """Extract well-formed sentences from text"""
        if not self.is_bangla_text(text):
            return []
        
        # Clean text
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Split on sentence ending markers
        sentence_pattern = r'[^।!?]+[।!?]'
        sentences = re.findall(sentence_pattern, text)
        
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            
            # Check minimum length and ensure it contains Bengali characters
            words = sentence.split()
            if (len(words) >= 5 and len(words) <= 50 and 
                self.is_bangla_text(sentence) and
                not re.search(r'[0-9]{4,}', sentence)):  # Avoid phone numbers, years etc.
                
                # Clean punctuation spacing
                sentence = re.sub(r'\s+([।!?,.;:])', r'\1', sentence)
                sentence = re.sub(r'([।!?,.;:])\s*', r'\1 ', sentence).strip()
                
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def scrape_url(self, url: str, site_type: str = "general") -> List[str]:
        """Scrape content from a single URL"""
        sentences = []
        
        try:
            headers = self.get_random_headers()
            response = self.session.get(
                url, 
                headers=headers, 
                timeout=self.config.get("request_timeout", 10),
                allow_redirects=True
            )
            
            if response.status_code != 200:
                return sentences
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'form']):
                element.decompose()
            
            # Try different content selectors based on site type
            selectors = self.content_selectors.get(site_type, self.content_selectors["general"])
            
            for selector in selectors:
                elements = soup.select(selector)
                for element in elements:
                    text = element.get_text().strip()
                    if text and len(text) > 30:
                        extracted_sentences = self.extract_sentences_from_text(text)
                        sentences.extend(extracted_sentences)
            
            # If no content found with specific selectors, try general approach
            if not sentences:
                all_paragraphs = soup.find_all('p')
                for p in all_paragraphs:
                    text = p.get_text().strip()
                    if text and len(text) > 30:
                        extracted_sentences = self.extract_sentences_from_text(text)
                        sentences.extend(extracted_sentences)
            
            # Add delay to be respectful
            time.sleep(random.uniform(0.5, 2.0))
            
        except Exception as e:
            logger.warning(f"Error scraping {url}: {e}")
        
        return sentences
    
    def get_article_links_from_site(self, base_url: str, max_links: int = 50) -> List[str]:
        """Get article links from a news site or blog"""
        links = []
        
        try:
            headers = self.get_random_headers()
            response = self.session.get(base_url, headers=headers, timeout=10)
            
            if response.status_code != 200:
                return links
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find all article links
            link_selectors = [
                'a[href*="/news/"]', 'a[href*="/article/"]', 'a[href*="/post/"]',
                'a[href*="/story/"]', 'a[href*="/blog/"]', '.article-title a',
                '.news-title a', '.post-title a', '.story-title a', 'h2 a', 'h3 a'
            ]
            
            for selector in link_selectors:
                elements = soup.select(selector)
                for element in elements:
                    href = element.get('href')
                    if href:
                        # Convert relative URLs to absolute
                        full_url = urljoin(base_url, href)
                        
                        # Filter out unwanted links
                        if (full_url not in links and 
                            not any(x in full_url.lower() for x in ['video', 'photo', 'gallery', 'podcast', 'live']) and
                            len(links) < max_links):
                            links.append(full_url)
            
        except Exception as e:
            logger.warning(f"Error getting links from {base_url}: {e}")
        
        return links[:max_links]
    
    def scrape_rss_feeds(self) -> List[str]:
        """Scrape article URLs from RSS feeds"""
        sentences = []
        
        logger.info(f"Scraping {len(self.rss_feeds)} RSS feeds...")
        
        for feed_url in self.rss_feeds:
            try:
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries[:10]:  # Limit per feed
                    if hasattr(entry, 'link'):
                        article_sentences = self.scrape_url(entry.link, "news")
                        sentences.extend(article_sentences)
                        
                        # Add delay between articles
                        time.sleep(1)
                
            except Exception as e:
                logger.warning(f"Error scraping RSS feed {feed_url}: {e}")
        
        return sentences
    
    def scrape_wikipedia_extensively(self, num_articles: int = 200) -> List[str]:
        """Extensively scrape Bengali Wikipedia"""
        wikipedia.set_lang('bn')
        sentences = []
        
        logger.info(f"Scraping {num_articles} Bengali Wikipedia articles...")
        
        # Get articles from different categories
        categories = [
            "বাংলাদেশ", "ভারত", "সাহিত্য", "ইতিহাস", "বিজ্ঞান", 
            "প্রযুক্তি", "রাজনীতি", "অর্থনীতি", "সংস্কৃতি", "ভূগোল"
        ]
        
        articles_per_category = num_articles // len(categories)
        
        for category in categories:
            try:
                # Search for articles in each category
                search_results = wikipedia.search(category, results=articles_per_category)
                
                for title in search_results:
                    try:
                        page = wikipedia.page(title)
                        article_sentences = self.extract_sentences_from_text(page.content)
                        sentences.extend(article_sentences)
                        
                        time.sleep(0.1)  # Be respectful
                        
                    except Exception as e:
                        logger.warning(f"Error fetching Wikipedia article {title}: {e}")
                        
            except Exception as e:
                logger.warning(f"Error searching Wikipedia category {category}: {e}")
        
        logger.info(f"Extracted {len(sentences)} sentences from Wikipedia")
        return sentences
    
    def scrape_news_portals_extensively(self, max_articles_per_site: int = 20) -> List[str]:
        """Extensively scrape ALL Bengali news portals"""
        sentences = []
        
        # Combine all news sources
        all_news_sources = (
            self.bangla_sources['major_news_portals'] + 
            self.bangla_sources['international_bangla_news']
        )
        
        logger.info(f"Extensively scraping {len(all_news_sources)} news portals...")
        
        for site_url in tqdm(all_news_sources, desc="Scraping ALL news sites"):
            try:
                # Get article links from the main page
                article_links = self.get_article_links_from_site(site_url, max_articles_per_site)
                
                # Scrape each article
                for link in article_links[:max_articles_per_site]:
                    article_sentences = self.scrape_url(link, "news")
                    sentences.extend(article_sentences)
                    
                    # Add delay between articles
                    time.sleep(random.uniform(1, 3))
                
            except Exception as e:
                logger.warning(f"Error scraping news portal {site_url}: {e}")
        
        logger.info(f"Extracted {len(sentences)} sentences from ALL news portals")
        return sentences
    
    def scrape_blogs_and_forums(self) -> List[str]:
        """Scrape Bengali blogs and forums"""
        sentences = []
        
        logger.info("Scraping Bengali blogs and forums...")
        
        for blog_url in self.bangla_sources['blogs_forums']:
            try:
                # Get blog post links
                post_links = self.get_article_links_from_site(blog_url, 30)
                
                for link in post_links:
                    post_sentences = self.scrape_url(link, "blog")
                    sentences.extend(post_sentences)
                    
                    time.sleep(2)  # Be more respectful with blogs
                
            except Exception as e:
                logger.warning(f"Error scraping blog {blog_url}: {e}")
        
        logger.info(f"Extracted {len(sentences)} sentences from blogs and forums")
        return sentences
    
    def scrape_educational_content(self) -> List[str]:
        """Scrape educational and reference content"""
        sentences = []
        
        logger.info("Scraping educational content...")
        
        for edu_url in self.bangla_sources['educational']:
            try:
                if 'wikipedia' in edu_url:
                    continue  # Skip, we handle Wikipedia separately
                
                content_sentences = self.scrape_url(edu_url, "general")
                sentences.extend(content_sentences)
                
                time.sleep(2)
                
            except Exception as e:
                logger.warning(f"Error scraping educational site {edu_url}: {e}")
        
        logger.info(f"Extracted {len(sentences)} sentences from educational content")
        return sentences
    
    def discover_bangla_sites_via_search(self, max_sites: int = 100) -> List[str]:
        """
        Discover Bangla websites through search engines
        """
        discovered_sites = []
        
        search_queries = [
            "বাংলা ওয়েবসাইট",
            "বাংলাদেশের সংবাদ",
            "বাংলা ব্লগ",
            "বাংলা সাহিত্য",
            "বাংলা শিক্ষা",
            "site:.bd বাংলা",
            "site:.com বাংলাদেশ",
            "বাংলা অনলাইন"
        ]
        
        # This would require implementing search engine scraping
        # For now, return predefined discovered sites
        discovered_sites = [
            "https://www.ajkerpatrika.com/",
            "https://www.shirshanews.com/",
            "https://www.dainikduniya.com/",
            "https://www.protidiner.com/",
            "https://www.natokpala.com/",
            "https://www.charbak.com/",
            "https://www.shorol.com/",
            "https://www.nirbachito.com/"
        ]
        
        logger.info(f"Discovered {len(discovered_sites)} additional Bangla sites")
        return discovered_sites[:max_sites]
    
    def scrape_archive_org_bangla(self) -> List[str]:
        """
        Scrape historical Bangla content from Internet Archive
        """
        sentences = []
        
        # Internet Archive Wayback Machine for Bangla sites
        archive_urls = [
            "https://web.archive.org/web/*/prothomalo.com/*",
            "https://web.archive.org/web/*/kalerkantho.com/*",
            "https://web.archive.org/web/*/bn.wikipedia.org/*"
        ]
        
        logger.info("Scraping historical content from Internet Archive...")
        
        # Implementation would access archived pages
        # For now, simulate with placeholder
        logger.info("Internet Archive scraping - placeholder implementation")
        
        return sentences
    
    def scrape_social_media_bangla(self) -> List[str]:
        """
        Scrape Bangla content from social media platforms
        Note: Requires API access for most platforms
        """
        sentences = []
        
        # Social media platforms with Bangla content
        social_platforms = {
            "facebook": "https://www.facebook.com/public/posts?q=বাংলা",
            "twitter": "https://twitter.com/search?q=বাংলা",
            "youtube": "https://www.youtube.com/results?search_query=বাংলা",
            "reddit": "https://www.reddit.com/r/bangladesh/",
            "quora": "https://bn.quora.com/"
        }
        
        # For Reddit (publicly accessible)
        try:
            reddit_url = "https://www.reddit.com/r/bangladesh/.json"
            headers = self.get_random_headers()
            response = self.session.get(reddit_url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                for post in data.get('data', {}).get('children', []):
                    title = post.get('data', {}).get('title', '')
                    selftext = post.get('data', {}).get('selftext', '')
                    
                    for text in [title, selftext]:
                        if text and self.is_bangla_text(text):
                            extracted = self.extract_sentences_from_text(text)
                            sentences.extend(extracted)
            
            time.sleep(2)
            
        except Exception as e:
            logger.warning(f"Error scraping Reddit: {e}")
        
        logger.info(f"Extracted {len(sentences)} sentences from social media")
        return sentences
    
    def scrape_academic_repositories(self) -> List[str]:
        """
        Scrape Bangla content from academic repositories
        """
        sentences = []
        
        academic_sources = [
            "https://www.researchgate.net/search/publication?q=bengali",
            "https://scholar.google.com/scholar?q=bengali+language",
            "https://arxiv.org/search/?query=bengali&searchtype=all",
            "https://www.academia.edu/search?q=bengali",
            "https://www.jstor.org/action/doBasicSearch?Query=bengali",
            "https://dspace.library.iitb.ac.in/jspui/browse?type=subject&value=Bengali"
        ]
        
        logger.info("Scraping academic repositories for Bangla content...")
        
        for source in academic_sources:
            try:
                # For now, simulate academic scraping
                # Real implementation would parse academic papers
                time.sleep(1)
                
            except Exception as e:
                logger.warning(f"Error scraping academic source {source}: {e}")
        
        logger.info(f"Extracted {len(sentences)} sentences from academic sources")
        return sentences
    
    def scrape_government_documents(self) -> List[str]:
        """
        Scrape Bangla content from government documents and portals
        """
        sentences = []
        
        govt_sources = [
            "https://www.bangladesh.gov.bd/",
            "https://www.cabinet.gov.bd/",
            "https://www.pmo.gov.bd/",
            "https://www.parliament.gov.bd/",
            "https://www.mof.gov.bd/",
            "https://www.mopa.gov.bd/",
            "https://www.nctb.gov.bd/",
            "https://www.education.gov.bd/"
        ]
        
        logger.info("Scraping government documents and portals...")
        
        for govt_url in govt_sources:
            try:
                govt_sentences = self.scrape_url(govt_url, "general")
                sentences.extend(govt_sentences)
                time.sleep(2)  # Be respectful with government sites
                
            except Exception as e:
                logger.warning(f"Error scraping government site {govt_url}: {e}")
        
        logger.info(f"Extracted {len(sentences)} sentences from government sources")
        return sentences
    
    def scrape_multimedia_transcripts(self) -> List[str]:
        """
        Scrape transcripts from Bangla multimedia content
        """
        sentences = []
        
        multimedia_sources = [
            "https://www.youtube.com/results?search_query=বাংলা+সংবাদ",
            "https://www.youtube.com/results?search_query=বাংলা+নাটক",
            "https://www.youtube.com/results?search_query=বাংলা+গান",
            "https://www.chorki.com/",
            "https://www.hoichoi.tv/",
            "https://www.bongodorshon.com/"
        ]
        
        logger.info("Scraping multimedia transcripts and captions...")
        
        # For now, simulate multimedia transcript extraction
        # Real implementation would use YouTube API, video processing libraries
        logger.info("Multimedia transcript scraping - placeholder implementation")
        
        return sentences
    
    def scrape_all_internet_sources(self) -> List[str]:
        """
        MEGA METHOD: Scrape from ALL possible internet sources
        """
        all_sentences = []
        
        logger.info("🌐 STARTING COMPREHENSIVE INTERNET-WIDE BANGLA DATA COLLECTION")
        logger.info("=" * 80)
        
        # 1. All predefined source categories
        for category, urls in self.bangla_sources.items():
            logger.info(f"📂 Scraping {category} ({len(urls)} sources)...")
            try:
                for url in urls:
                    sentences = self.scrape_url(url, "general")
                    all_sentences.extend(sentences)
                    time.sleep(random.uniform(1, 3))
                logger.info(f"✅ {category}: collected sentences")
            except Exception as e:
                logger.error(f"❌ Error in {category}: {e}")
        
        # 2. Comprehensive RSS feeds
        logger.info("📡 Scraping ALL RSS feeds...")
        try:
            rss_sentences = self.scrape_rss_feeds()
            all_sentences.extend(rss_sentences)
            logger.info(f"✅ RSS feeds: {len(rss_sentences)} sentences")
        except Exception as e:
            logger.error(f"❌ RSS error: {e}")
        
        # 3. Discover new sites
        logger.info("🔍 Discovering additional Bangla sites...")
        try:
            discovered_sites = self.discover_bangla_sites_via_search()
            for site in discovered_sites:
                sentences = self.scrape_url(site, "general")
                all_sentences.extend(sentences)
                time.sleep(2)
            logger.info(f"✅ Discovered sites: {len(discovered_sites)} sites scraped")
        except Exception as e:
            logger.error(f"❌ Discovery error: {e}")
        
        # 4. Social media content
        logger.info("📱 Scraping social media content...")
        try:
            social_sentences = self.scrape_social_media_bangla()
            all_sentences.extend(social_sentences)
            logger.info(f"✅ Social media: {len(social_sentences)} sentences")
        except Exception as e:
            logger.error(f"❌ Social media error: {e}")
        
        # 5. Academic repositories
        logger.info("🎓 Scraping academic repositories...")
        try:
            academic_sentences = self.scrape_academic_repositories()
            all_sentences.extend(academic_sentences)
            logger.info(f"✅ Academic: {len(academic_sentences)} sentences")
        except Exception as e:
            logger.error(f"❌ Academic error: {e}")
        
        # 6. Government documents
        logger.info("🏛️ Scraping government documents...")
        try:
            govt_sentences = self.scrape_government_documents()
            all_sentences.extend(govt_sentences)
            logger.info(f"✅ Government: {len(govt_sentences)} sentences")
        except Exception as e:
            logger.error(f"❌ Government error: {e}")
        
        # 7. Internet Archive
        logger.info("📚 Scraping Internet Archive...")
        try:
            archive_sentences = self.scrape_archive_org_bangla()
            all_sentences.extend(archive_sentences)
            logger.info(f"✅ Archive: {len(archive_sentences)} sentences")
        except Exception as e:
            logger.error(f"❌ Archive error: {e}")
        
        # 8. Multimedia transcripts
        logger.info("🎥 Scraping multimedia transcripts...")
        try:
            multimedia_sentences = self.scrape_multimedia_transcripts()
            all_sentences.extend(multimedia_sentences)
            logger.info(f"✅ Multimedia: {len(multimedia_sentences)} sentences")
        except Exception as e:
            logger.error(f"❌ Multimedia error: {e}")
        
        return all_sentences
    
    def scrape_comprehensive_bangla_data(self, 
                                       wikipedia_articles: int = 200,
                                       news_articles_per_site: int = 20,
                                       include_blogs: bool = True,
                                       include_educational: bool = True,
                                       include_all_internet: bool = False) -> List[str]:
        """
        Comprehensive scraping of Bangla data from the internet
        
        Args:
            wikipedia_articles: Number of Wikipedia articles to scrape
            news_articles_per_site: Max articles per news site
            include_blogs: Whether to include blog content
            include_educational: Whether to include educational content
            include_all_internet: Whether to scrape from ALL internet sources
            
        Returns:
            List of extracted sentences
        """
        
        if include_all_internet:
            logger.info("🚀 ACTIVATING MAXIMUM INTERNET SCRAPING MODE")
            return self.scrape_all_internet_sources()
        
        # Standard comprehensive scraping
        all_sentences = []
        
        logger.info("Starting comprehensive Bangla data collection from the internet...")
        
        # 1. Wikipedia (most reliable source)
        try:
            wiki_sentences = self.scrape_wikipedia_extensively(wikipedia_articles)
            all_sentences.extend(wiki_sentences)
            logger.info(f"Wikipedia: {len(wiki_sentences)} sentences")
        except Exception as e:
            logger.error(f"Error in Wikipedia scraping: {e}")
        
        # 2. News portals (formal, recent content)
        try:
            news_sentences = self.scrape_news_portals_extensively(news_articles_per_site)
            all_sentences.extend(news_sentences)
            logger.info(f"News portals: {len(news_sentences)} sentences")
        except Exception as e:
            logger.error(f"Error in news portal scraping: {e}")
        
        # 3. RSS feeds (recent news)
        try:
            rss_sentences = self.scrape_rss_feeds()
            all_sentences.extend(rss_sentences)
            logger.info(f"RSS feeds: {len(rss_sentences)} sentences")
        except Exception as e:
            logger.error(f"Error in RSS scraping: {e}")
        
        # 4. Blogs and forums (informal content)
        if include_blogs:
            try:
                blog_sentences = self.scrape_blogs_and_forums()
                all_sentences.extend(blog_sentences)
                logger.info(f"Blogs and forums: {len(blog_sentences)} sentences")
            except Exception as e:
                logger.error(f"Error in blog scraping: {e}")
        
        # 5. Educational content
        if include_educational:
            try:
                edu_sentences = self.scrape_educational_content()
                all_sentences.extend(edu_sentences)
                logger.info(f"Educational content: {len(edu_sentences)} sentences")
            except Exception as e:
                logger.error(f"Error in educational content scraping: {e}")
        
        # Remove duplicates while preserving order
        unique_sentences = []
        seen = set()
        for sentence in all_sentences:
            if sentence not in seen:
                seen.add(sentence)
                unique_sentences.append(sentence)
        
        logger.info(f"Total unique sentences collected: {len(unique_sentences)}")
        logger.info(f"Duplicates removed: {len(all_sentences) - len(unique_sentences)}")
        
        return unique_sentences
    
    def save_scraped_data(self, sentences: List[str], output_file: str = "scraped_bangla_data.json"):
        """Save scraped data to file"""
        data = {
            "total_sentences": len(sentences),
            "collection_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "sentences": sentences
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(sentences)} sentences to {output_file}")

# Example usage
if __name__ == "__main__":
    scraper = BanglaWebScraper()
    
    # Comprehensive scraping
    sentences = scraper.scrape_comprehensive_bangla_data(
        wikipedia_articles=100,
        news_articles_per_site=15,
        include_blogs=True,
        include_educational=True
    )
    
    # Save data
    scraper.save_scraped_data(sentences)
    
    print(f"Successfully collected {len(sentences)} Bangla sentences from the internet!")
