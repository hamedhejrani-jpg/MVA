import unittest
from app.main import InstallationRequest, validation_issues, ai_configuration


class MvaValidationTests(unittest.TestCase):
    def test_valid_tv_install_with_fixed_and_mobile(self):
        item = InstallationRequest(
            full_name="کاربر آزمایشی",
            national_id="0012345678",
            phone="02112345678",
            mobile="09123456789",
            product="تلویزیون",
            serial_number="TEST-SERIAL",
            install_type="زمینی",
            consent_confirmed=True,
        )
        self.assertEqual(validation_issues(item), [])

    def test_requires_two_mobiles_without_landline(self):
        item = InstallationRequest(
            full_name="کاربر آزمایشی",
            national_id="0012345678",
            mobile="09123456789",
            product="تلویزیون",
            serial_number="TEST-SERIAL",
            install_type="دیواری",
        )
        self.assertTrue(any(x["field"] == "phone" for x in validation_issues(item)))

    def test_rejects_non_tv(self):
        item = InstallationRequest(
            full_name="کاربر آزمایشی",
            national_id="0012345678",
            phone="02112345678",
            mobile="09123456789",
            product="کولر",
            serial_number="TEST-SERIAL",
            install_type="زمینی",
        )
        self.assertTrue(any(x["field"] == "product" for x in validation_issues(item)))

    def test_ai_configuration_reports_configured_flag(self):
        cfg = ai_configuration()
        self.assertIn("configured", cfg)
        self.assertIn("enabled", cfg)


if __name__ == "__main__":
    unittest.main()
