"""检查 PASM v0 是否可脱离旧 PSMA 项目运行。"""

from pathlib import Path
import unittest


class 项目独立性测试(unittest.TestCase):
    """验证新项目不再导入旧实现。"""

    def test_源码目录存在且不依赖旧模块(self):
        """新项目的源码不得导入 PSMA.pasm。"""
        project_root = Path(__file__).resolve().parents[1]
        package_root = project_root / "pasm_v0"
        self.assertTrue(package_root.is_dir(), "缺少独立的 pasm_v0 源码目录")

        for source_file in package_root.glob("*.py"):
            source_text = source_file.read_text(encoding="utf-8")
            self.assertNotIn("PSMA.pasm", source_text, source_file.name)

    def test_测试目录不依赖旧项目包名(self):
        """迁入的 v0 测试必须直接导入独立项目包。"""
        test_root = Path(__file__).resolve().parent
        for test_file in test_root.glob("test_*.py"):
            if test_file == Path(__file__):
                continue
            source_text = test_file.read_text(encoding="utf-8")
            self.assertNotIn("PSMA.pasm_v0", source_text, test_file.name)


if __name__ == "__main__":
    unittest.main()
