from icrawler.builtin import BingImageCrawler

classes = {
    'Delfines': 'dolphin underwater',
    'Pulpos': 'octopus underwater realistic photo',
    'Tortugas': 'sea turtle underwater realistic photo',
}

def download_images():
    for folder, keyword in classes.items():
        crawler = BingImageCrawler(
            storage={'root_dir': f'dataset/{folder}'}
        )
        crawler.crawl(keyword=keyword, max_num=900)

if __name__ == "__main__":
    download_images()