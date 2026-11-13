'use strict';

// Browser-only illustration. No application requests, storage or order writes.
(() => {
  const samples = {
    daily: { label: 'A single illustrative day', rows: [[1, 2, 160], [1, 1, 80], [1, 1, 80]] },
    weekly: { label: 'Saturday through Friday', rows: [[2, 4, 320], [1, 2, 160], [2, 3, 240]] },
    monthly: { label: 'An illustrative calendar month', rows: [[8, 14, 1120], [6, 9, 720], [7, 12, 960]] },
  };
  const buttons = document.querySelectorAll('[data-period]');
  const rows = document.querySelectorAll('#example-rows tr');
  const caption = document.getElementById('period-caption');
  if (!caption || rows.length !== 3) return;
  const money = (amount) => `₹${amount.toLocaleString('en-IN')}`;
  buttons.forEach((button) => button.addEventListener('click', () => {
    const sample = samples[button.dataset.period];
    if (!sample) return;
    buttons.forEach((item) => item.setAttribute('aria-pressed', String(item === button)));
    rows.forEach((row, index) => {
      const cells = row.querySelectorAll('td');
      cells[0].textContent = String(sample.rows[index][0]);
      cells[1].textContent = String(sample.rows[index][1]);
      cells[2].textContent = money(sample.rows[index][2]);
    });
    caption.textContent = sample.label;
    document.getElementById('total-records').textContent = String(sample.rows.reduce((sum, row) => sum + row[0], 0));
    document.getElementById('total-quantity').textContent = String(sample.rows.reduce((sum, row) => sum + row[1], 0));
    document.getElementById('total-amount').textContent = money(sample.rows.reduce((sum, row) => sum + row[2], 0));
  }));
})();
