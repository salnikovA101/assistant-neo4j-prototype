#!/usr/bin/env python3
"""Interactive read-only inspection of the kefir S3 bucket. No saved credentials."""

import argparse
import getpass
import json
import re
import sys
import tempfile
import ssl
import urllib.error
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', default='', help='Show only keys starting with this prefix')
    parser.add_argument('--check-connection', action='store_true', help='Check HTTPS without credentials')
    parser.add_argument('--ca-bundle', help='Trusted CA PEM file; HTTPS verification stays enabled')
    parser.add_argument('--region', default='ru-3', help='S3 signing region (default: ru-3)')
    args = parser.parse_args()
    if args.ca_bundle:
        try:
            ssl.create_default_context(cafile=args.ca_bundle)
        except (OSError, ssl.SSLError):
            parser.exit(1, 'Не удалось прочитать CA-файл в формате PEM.\n')
    if args.check_connection:
        try:
            from botocore.httpsession import get_cert_path
            context = ssl.create_default_context(cafile=args.ca_bundle or get_cert_path(True))
            with urllib.request.urlopen('https://s3.ru-3.storage.selcloud.ru', context=context, timeout=15):
                print('HTTPS работает. Доступ к бакету не проверялся.')
        except urllib.error.HTTPError as exc:
            print(f'HTTPS работает (HTTP {exc.code}). Доступ к бакету не проверялся.')
        except urllib.error.URLError as exc:
            print(f'Сбой соединения без ключей: {exc.reason}', file=sys.stderr)
            return 1
        except (ImportError, OSError) as exc:
            print(f'Проверка не завершена: {type(exc).__name__}', file=sys.stderr)
            return 1
        return 0
    if not sys.stdin.isatty():
        parser.exit(1, 'Run in your own interactive terminal; credentials must be entered secretly.\n')
    try:
        import boto3
        from botocore.config import Config
        from botocore.exceptions import BotoCoreError, ClientError, SSLError
    except ImportError:
        parser.exit(1, 'Install boto3 in a separate virtual environment first (see instructions).\n')

    access = getpass.getpass('Access key (скрытый ввод): ').strip()
    secret = getpass.getpass('Secret key (скрытый ввод): ').strip()
    if not access or not secret:
        parser.exit(1, 'Оба ключа обязательны.\n')
    client = boto3.client(
        's3', endpoint_url='https://s3.ru-3.storage.selcloud.ru',
        region_name=args.region, aws_access_key_id=access, aws_secret_access_key=secret,
        verify=args.ca_bundle or True,
        config=Config(signature_version='s3v4', s3={'addressing_style': 'path'},
                      connect_timeout=10, read_timeout=30, retries={'max_attempts': 2}),
    )
    limit = 50 * 1024 * 1024
    token = None
    try:
        while True:
            request = dict(Bucket='kefir', Prefix=args.prefix, MaxKeys=100)
            if token:
                request['ContinuationToken'] = token
            page = client.list_objects_v2(**request)
            objects = page.get('Contents', [])
            if not objects:
                print('Объектов с этим префиксом нет.')
                return
            for number, item in enumerate(objects, 1):
                # JSON escaping prevents control characters in remote keys reaching the terminal.
                print(f'{number:3}  {item["Size"] / 1024 / 1024:8.2f} МБ  '
                      f'{json.dumps(item["Key"], ensure_ascii=True)}')
            print('Это одна страница списка, не полный подсчёт содержимого бакета.')
            while True:
                choice = input('Номер PDF — скачать; n — следующая страница; q — выход: ').strip()
                if choice.lower() == 'q':
                    return
                if choice.lower() == 'n':
                    token = page.get('NextContinuationToken')
                    if token:
                        break
                    print('Это последняя страница.')
                    continue
                if not choice.isdigit() or not 1 <= int(choice) <= len(objects):
                    print('Введите номер из списка, n или q.')
                    continue
                item = objects[int(choice) - 1]
                if not item['Key'].lower().endswith('.pdf'):
                    print('Скачивание разрешено только для файлов с расширением .pdf.')
                    continue
                if item['Size'] > limit:
                    print('Файл превышает лимит 50 МБ.')
                    continue
                response = client.get_object(Bucket='kefir', Key=item['Key'])
                body = response['Body']
                output = None
                try:
                    if response['ContentLength'] > limit:
                        print('Текущий размер файла превышает лимит 50 МБ.')
                        continue
                    first = body.read(1024)
                    if b'%PDF-' not in first:
                        print('В начале файла нет сигнатуры PDF; файл не сохранён.')
                        continue
                    folder = Path(__file__).resolve().parents[1] / 'data' / 's3-preview'
                    folder.mkdir(parents=True, exist_ok=True)
                    # Use a generated local name: remote keys cannot escape the folder or overwrite files.
                    with tempfile.NamedTemporaryFile(dir=folder, prefix='article-', suffix='.pdf', delete=False) as f:
                        output = Path(f.name)
                        f.write(first)
                        size = len(first)
                        while True:
                            chunk = body.read(min(1024 * 1024, limit - size + 1))
                            if not chunk:
                                break
                            size += len(chunk)
                            if size > limit:
                                raise ValueError('Размер скачивания превысил 50 МБ')
                            f.write(chunk)
                    print(f'Сохранено: {output}')
                except BaseException:
                    if output:
                        output.unlink(missing_ok=True)
                    raise
                finally:
                    body.close()
    except ClientError as exc:
        error = exc.response.get('Error', {})
        def safe(value):
            value = str(value).replace(access, '[REDACTED]').replace(secret, '[REDACTED]')
            value = re.sub(r'(?i)(credential|signature|authorization)\s*[=:]\s*[^\s,;]+',
                           r'\1=[REDACTED]', value)
            value = re.sub(r'\b[0-9a-fA-F]{32,}\b', '[REDACTED]', value)
            return json.dumps(value[:1500], ensure_ascii=True)
        print('Ошибка S3:', safe(error.get('Code', 'Unknown')), file=sys.stderr)
        print('Причина:', safe(error.get('Message', 'Не указана')), file=sys.stderr)
        headers = exc.response.get('ResponseMetadata', {}).get('HTTPHeaders', {})
        region = error.get('Region') or headers.get('x-amz-bucket-region')
        if region:
            print('Регион от сервера:', safe(region), file=sys.stderr)
        if error.get('Code') == 'AuthorizationHeaderMalformed':
            print('Сервер отклонил формат авторизации или параметры подписи. '
                  'Если он указал ожидаемый регион, передайте его через --region.', file=sys.stderr)
        return 1
    except SSLError:
        print('Сбой HTTPS до получения ответа S3. Запустите с --check-connection без ключей. '
              'Не отключайте проверку сертификата.', file=sys.stderr)
        return 1
    except (BotoCoreError, OSError, ValueError) as exc:
        # Do not dump SDK exceptions, request headers, or credentials.
        print(f'Не удалось завершить запрос ({type(exc).__name__}).', file=sys.stderr)
        return 1
    finally:
        client.close()


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (KeyboardInterrupt, EOFError):
        print('\nВыход.')
