struct offset
{
	size_t upper;
	size_t lower;
};


class view final
{
	uint32_t n_dims_;
	char data_size_{};

	uint32_t* shape_;
	size_t* size_;
	size_t* stride_;

	uint32_t n_offsets_;
	offset** offset_;
public:
	view(const uint32_t* shape, uint32_t dims, char data_size);
	view(uint32_t* shape, uint32_t dims, char data_size);

	~view();
	view(const view&v);
	view(view&& v) noexcept;
	view& operator=(view&& v) noexcept;
	[[nodiscard]] uint32_t ndims() const;
	[[nodiscard]] size_t size(uint32_t idx = 0) const;
	[[nodiscard]] uint32_t shape(uint32_t idx = 0) const;
	[[nodiscard]] size_t bytes_length() const;
	[[nodiscard]] size_t count(uint32_t start_axis = 0) const;
	[[nodiscard]] size_t count(const uint32_t start_axis, uint32_t end_axis=-1) const;
	view index(uint32_t idx, int dim);
	view reshape(const uint32_t* shape, uint32_t dims);

};
