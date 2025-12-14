# Đóng góp cho Hope V1.0.0

Chúng tôi rất mong bạn đóng góp cho dự án này và giúp nó trở nên tốt hơn nữa!
Là một người đóng góp, dưới đây là các hướng dẫn mà chúng tôi mong bạn tuân thủ:

- [Quy tắc ứng xử](#code-of-conduct)
- [Câu hỏi hoặc Vấn đề?](#question-or-problem)
- [Vấn đề và Lỗi](#found-a-bug)
- [Yêu cầu Tính năng](#missing-a-feature)
- [Hướng dẫn Gửi đóng góp](#submission-guidelines)
- [Quy tắc Lập trình](#coding-rules)
- [Hướng dẫn Viết Commit Message](#commit-message-format)
- [Ký CLA](#signing-the-cla)

## Quy tắc ứng xử

Hãy giúp chúng tôi giữ cho dự án này cởi mở và hòa nhập.
Vui lòng đọc và tuân theo [Quy tắc ứng xử](./CODE_OF_CONDUCT.md) của chúng tôi.

## Câu hỏi hoặc Vấn đề?

Đừng mở issue (vấn đề) cho các câu hỏi hỗ trợ chung vì chúng tôi muốn giữ GitHub issues cho các báo cáo lỗi và yêu cầu tính năng.
Thay vào đó, chúng tôi khuyên bạn nên hỏi trực tiếp qua [email](mailto:trananhquan1009@gmail.com) để đặt các câu hỏi liên quan đến hỗ trợ.

Discord là nơi tốt hơn nhiều để đặt câu hỏi vì:

- Có cộng đồng sẵn sàng giúp đỡ.
- Các câu hỏi và câu trả lời luôn có sẵn để mọi người cùng xem.

Chúng tôi sẽ đóng tất cả các issue mang tính chất yêu cầu hỗ trợ chung và hướng dẫn mọi người qua Discord.

## Phát hiện Lỗi?

Nếu bạn tìm thấy lỗi trong mã nguồn, bạn có thể giúp chúng tôi bằng cách [gửi một issue](#gửi-một-issue) đến [Kho lưu trữ GitHub](https://github.com/Hope-Defensive/Hope-AD/issues) của chúng tôi.
Tuyệt vời hơn nữa, bạn có thể [gửi một Pull Request](#gửi-một-pull-request-pr) kèm theo bản sửa lỗi.

## Thiếu Tính năng?

Bạn có thể _yêu cầu_ một tính năng mới bằng cách [gửi một issue](#gửi-một-issue) đến Kho lưu trữ GitHub của chúng tôi.
Nếu bạn muốn _triển khai_ một tính năng mới, vui lòng xem xét quy mô của thay đổi:

- Đối với một **Tính năng Lớn**, trước tiên hãy mở một issue và phác thảo đề xuất của bạn để có thể thảo luận.
- **Tính năng Nhỏ** có thể được thực hiện và [gửi trực tiếp dưới dạng Pull Request](#gửi-một-pull-request-pr).

## Hướng dẫn Gửi đóng góp

### Gửi một Issue

Trước khi bạn gửi một issue, vui lòng tìm kiếm trong trình theo dõi vấn đề (issue tracker). Có thể đã có một issue cho vấn đề của bạn và cuộc thảo luận ở đó có thể cung cấp cho bạn giải pháp thay thế.

Chúng tôi muốn sửa tất cả các vấn đề càng sớm càng tốt, nhưng trước khi sửa lỗi, chúng tôi cần tái hiện và xác nhận nó. Để tái hiện lỗi, chúng tôi yêu cầu bạn cung cấp một bản tái hiện tối thiểu (minimal reproduction).

Bạn có thể tạo issue mới bằng cách chọn từ [các mẫu issue mới](https://github.com/Hope-Defensive/Hope-AD/issues/new/choose) của chúng tôi và điền vào mẫu.

### Gửi một Pull Request (PR)

Trước khi bạn gửi Pull Request (PR), hãy xem xét các hướng dẫn sau:

1. Tìm kiếm [GitHub PRs](https://github.com/Hope-Defensive/Hope-AD/pulls) để xem có PR nào đang mở hoặc đã đóng liên quan đến nội dung bạn gửi không.
2. Đảm bảo rằng có một issue mô tả vấn đề bạn đang sửa, hoặc tài liệu hóa thiết kế cho tính năng bạn muốn thêm.
3. Vui lòng ký [Thỏa thuận Cấp phép Người đóng góp (CLA)](#ký-cla) của chúng tôi trước khi gửi PR.
4. [Fork](https://docs.github.com/en/github/getting-started-with-github/fork-a-repo) kho lưu trữ.
5. Trong kho lưu trữ đã fork của bạn, hãy thực hiện các thay đổi trong một nhánh git mới:

   ```sh
   git checkout -b my-fix-branch main
   ```

6. Tạo bản vá (patch) của bạn.
7. Tuân theo [Quy tắc Lập trình](https://www.google.com/search?q=%23quy-t%E1%BA%AFc-l%E1%BA%ADp-tr%C3%ACnh) của chúng tôi.
8. Đảm bảo kiểm thử các thay đổi của bạn.
9. Commit các thay đổi của bạn bằng cách sử dụng thông điệp commit mô tả tuân theo [quy ước thông điệp commit](https://www.google.com/search?q=%23%C4%91%E1%BB%8Bnh-d%E1%BA%A1ng-commit-message) của chúng tôi.
   ```sh
   git commit --all
   ```
10. Push nhánh của bạn lên GitHub:
    ```sh
    git push origin my-fix-branch
    ```
11. Trên GitHub, gửi một pull request đến nhánh `main`.

#### Review Pull Request

Các nhà bảo trì có quyền không chấp nhận pull request từ các thành viên cộng đồng không tuân thủ các quy tắc ứng xử tốt. Hành vi như vậy bao gồm việc không tuân theo [Quy tắc ứng xử](https://www.google.com/search?q=./CODE_OF_CONDUCT.md).

##### Giải quyết phản hồi review

Nếu chúng tôi yêu cầu thay đổi thông qua review code, thì:

1.  Thực hiện các cập nhật cần thiết cho mã.
2.  Tạo một commit fixup và push lên kho lưu trữ GitHub của bạn:
    ```sh
    git commit --all --fixup HEAD
    git push
    ```

##### Cập nhật thông điệp commit

Người review có thể đề xuất thay đổi cho thông điệp commit. Để cập nhật thông điệp commit của commit cuối cùng trên nhánh của bạn:

1.  Checkout nhánh của bạn:
    ```sh
    git checkout my-fix-branch
    ```
2.  Amend (sửa đổi) commit cuối cùng và thay đổi thông điệp commit:
    ```sh
    git commit --amend
    ```
3.  Push lên kho lưu trữ GitHub của bạn:
    ```sh
    git push --force-with-lease
    ```

#### Sau khi pull request của bạn được merge

Sau khi pull request của bạn được merge, bạn có thể xóa nhánh của mình một cách an toàn và pull các thay đổi từ kho lưu trữ chính:

- Xóa nhánh remote trên GitHub thông qua giao diện web GitHub hoặc shell cục bộ của bạn:
  ```sh
  git push origin --delete my-fix-branch
  ```
- Checkout nhánh main:
  ```sh
  git checkout main -f
  ```
- Xóa nhánh cục bộ:
  ```sh
  git branch -D my-fix-branch
  ```
- Cập nhật nhánh main của bạn với phiên bản upstream mới nhất:
  ```sh
  git pull --ff upstream main
  ```

## Quy tắc Lập trình

Để đảm bảo tính nhất quán trong toàn bộ mã nguồn, hãy ghi nhớ các quy tắc này khi bạn làm việc:

- Tất cả các tính năng hoặc bản sửa lỗi **phải được kiểm thử**.
- Tất cả các phương thức API công khai **phải được tài liệu hóa**.

## Định dạng Commit Message

Chúng tôi có các quy tắc rất chính xác về cách định dạng các thông điệp commit Git. Định dạng này giúp **lịch sử commit dễ đọc hơn**.

Mỗi thông điệp commit bao gồm một **header** (tiêu đề), một **body** (phần thân), và một **footer** (chân trang).

```
<header>
<BLANK LINE>
<body>
<BLANK LINE>
<footer>
```

Phần `header` là bắt buộc và phải tuân theo định dạng:

```
<type>(<scope>): <short summary>
```

- **type**: build|ci|docs|feat|fix|perf|refactor|test
- **scope**: api|commands|config|entities|events|guards|i18n|services|utils|cli|\#{issue_id}|(trống)
- **summary**: mệnh lệnh thức, thì hiện tại, không có dấu chấm ở cuối

Phần `body` là bắt buộc đối với tất cả các commit ngoại trừ loại "docs". Khi hiện diện, nó phải dài ít nhất 20 ký tự và giải thích động lực cho sự thay đổi.

Phần `footer` là tùy chọn. Nó có thể chứa thông tin về các thay đổi phá vỡ (breaking changes), các phần bị loại bỏ (deprecations), hoặc tham chiếu đến các issue/PR.

### Revert commits (Hoàn tác commit)

Nếu commit hoàn tác một commit trước đó, nó nên bắt đầu bằng `revert: `, theo sau là header của commit bị hoàn tác.

## Ký CLA

Vui lòng ký Thỏa thuận Cấp phép Người đóng góp (CLA) của chúng tôi trước khi gửi pull request. Để bất kỳ thay đổi mã nào được chấp nhận, CLA phải được ký. Đây là một quá trình nhanh chóng\!

- Đối với cá nhân, chúng tôi có một biểu mẫu nhấp chuột đơn giản.
- Đối với các tập đoàn, chúng tôi sẽ cần bạn in, ký và gửi biểu mẫu.

Nếu bạn có nhiều hơn một tài khoản GitHub, hoặc nhiều địa chỉ email được liên kết với một tài khoản GitHub duy nhất, bạn phải ký CLA bằng địa chỉ email chính của tài khoản GitHub được sử dụng để tạo Git commit và gửi pull request.
