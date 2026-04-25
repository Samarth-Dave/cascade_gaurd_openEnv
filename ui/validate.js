const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const projectDir = 'D:\\MetaHack\\Round 2 (Finale)\\cascade-guard-core-main';
process.chdir(projectDir);

const results = {};

// Step 1: Check node_modules
console.log('=== Step 1: Check node_modules ===');
if (!fs.existsSync('node_modules')) {
  console.log('node_modules missing - running npm ci...');
  try {
    execSync('npm ci', { stdio: 'inherit' });
    results.npmCi = { success: true, code: 0 };
  } catch (e) {
    results.npmCi = { success: false, code: e.status };
  }
} else {
  console.log('node_modules already exists');
  results.npmCi = { success: true, code: 0, skipped: true };
}
console.log('');

// Step 2: npm run lint
console.log('=== Step 2: npm run lint ===');
try {
  execSync('npm run lint', { stdio: 'inherit' });
  results.lint = { success: true, code: 0 };
} catch (e) {
  results.lint = { success: false, code: e.status };
}
console.log('');

// Step 3: npm run test
console.log('=== Step 3: npm run test (with CI=true) ===');
process.env.CI = 'true';
try {
  execSync('npm run test', { stdio: 'inherit' });
  results.test = { success: true, code: 0 };
} catch (e) {
  results.test = { success: false, code: e.status };
}
console.log('');

// Step 4: npm run build
console.log('=== Step 4: npm run build ===');
try {
  execSync('npm run build', { stdio: 'inherit' });
  results.build = { success: true, code: 0 };
} catch (e) {
  results.build = { success: false, code: e.status };
}
console.log('');

// Summary
console.log('=== SUMMARY ===');
console.log(`npm ci: ${results.npmCi.skipped ? 'SKIPPED' : (results.npmCi.success ? 'PASS' : 'FAIL')} (Exit: ${results.npmCi.code})`);
console.log(`npm run lint: ${results.lint.success ? 'PASS' : 'FAIL'} (Exit: ${results.lint.code})`);
console.log(`npm run test: ${results.test.success ? 'PASS' : 'FAIL'} (Exit: ${results.test.code})`);
console.log(`npm run build: ${results.build.success ? 'PASS' : 'FAIL'} (Exit: ${results.build.code})`);
