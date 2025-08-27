import os
import argparse

def rename_images_in_directory(directory, prefix, start_number=1):
    """
    지정된 디렉토리의 모든 이미지 파일명을 순차적인 형식으로 변경합니다.
    예: image.jpg -> upper_000001.jpg

    Args:
        directory (str): 이미지 파일이 있는 디렉토리 경로.
        prefix (str): 새 파일명에 사용할 접두사 (예: 'upper').
        start_number (int): 순번의 시작 숫자.
    """
    supported_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    
    try:
        # 디렉토리의 모든 파일 목록을 가져옵니다.
        files = os.listdir(directory)
    except FileNotFoundError:
        print(f"오류: '{directory}' 디렉토리를 찾을 수 없습니다.")
        return

    # 이미지 파일만 필터링하고 일관된 순서를 위해 정렬합니다.
    image_files = sorted([f for f in files if os.path.splitext(f)[1].lower() in supported_extensions])

    if not image_files:
        print(f"'{directory}' 디렉토리에서 이미지 파일을 찾을 수 없습니다.")
        return

    print(f"'{directory}' 디렉토리에서 '{prefix}' 접두사로 변경할 {len(image_files)}개의 이미지를 찾았습니다.")
    
    count = start_number
    for filename in image_files:
        # 파일 확장자를 가져옵니다.
        extension = os.path.splitext(filename)[1]
        
        # 새 파일명을 만듭니다. (예: 'upper_000001.jpg')
        new_filename = f"{prefix}_{str(count).zfill(6)}{extension}"
        
        # 전체 경로를 가져옵니다.
        old_path = os.path.join(directory, filename)
        new_path = os.path.join(directory, new_filename)
        
        # 파일명을 변경합니다.
        try:
            os.rename(old_path, new_path)
            print(f"'{filename}' -> '{new_filename}' 으로 변경되었습니다.")
            count += 1
        except OSError as e:
            print(f"'{filename}' 파일명 변경 중 오류 발생: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="디렉토리 내의 이미지 파일명을 일괄 변경합니다.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("directory", type=str, help="이미지가 있는 디렉토리 경로.\n현재 디렉토리를 사용하려면 '.'을 입력하세요.")
    parser.add_argument("--prefix", type=str, required=True, help="새 파일명에 사용할 접두사 (예: 'upper').")
    parser.add_argument("--start", type=int, default=1, help="순번의 시작 숫자 (기본값: 1).")
    
    args = parser.parse_args()
    
    rename_images_in_directory(args.directory, args.prefix, args.start)
    print("\n파일명 변경이 완료되었습니다.")