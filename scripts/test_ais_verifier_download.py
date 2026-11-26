"""Exercise the scoring observer across retries of one guest file download."""
import unittest
from unittest.mock import Mock, patch

import requests

from ais_verifier import install_scoring_observer, read_guest_file, scoring_requests


class GuestFileTests(unittest.TestCase):
    base = 'http://guest'

    def download(self, path):
        # PythonController retries this read, then returns None if all attempts fail.
        for _ in range(2):
            try:
                return requests.post(self.base + '/file', data={'file_path': path}).content
            except requests.exceptions.ChunkedEncodingError:
                continue
        return None

    def test_successful_retry_within_one_download(self):
        responses = [requests.exceptions.ChunkedEncodingError(), Mock(status_code=200, content=b'data')]
        send = Mock(side_effect=responses)
        with patch.object(requests.Session, 'send', lambda *a, **kw: send(*a, **kw)):
            install_scoring_observer()
            with scoring_requests(self.base):
                self.assertEqual(read_guest_file(self.download, 'doc'), b'data')

    def test_another_download_cannot_clear_a_failed_read(self):
        responses = [requests.exceptions.ChunkedEncodingError()] * 2 + [Mock(status_code=200, content=b'')]
        send = Mock(side_effect=responses)
        with patch.object(requests.Session, 'send', lambda *a, **kw: send(*a, **kw)):
            install_scoring_observer()
            with self.assertRaisesRegex(RuntimeError, 'ChunkedEncodingError'), scoring_requests(self.base):
                self.assertIsNone(read_guest_file(self.download, 'doc'))
                self.assertEqual(read_guest_file(self.download, 'other'), b'')

    def test_action_error_still_invalidates_scoring(self):
        def read(path):
            with self.assertRaises(requests.exceptions.ChunkedEncodingError):
                requests.post(self.base + '/setup/execute')
            return b'data'

        send = Mock(side_effect=requests.exceptions.ChunkedEncodingError)
        with patch.object(requests.Session, 'send', lambda *a, **kw: send(*a, **kw)):
            install_scoring_observer()
            with self.assertRaisesRegex(RuntimeError, '/setup/execute'), scoring_requests(self.base):
                read_guest_file(read, 'doc')


if __name__ == '__main__':
    unittest.main()
